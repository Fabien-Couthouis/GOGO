#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre
#############################################################

import random
from players.playerInterface import PlayerInterface
from operator import itemgetter
import random
import copy
import numpy as np
from scipy.special import softmax
from goban import Goban, Stone, get_action_id
import math
import time
from players.alphazeroModel import AlphaZeroModel
from players.playerInterface import PlayerInterface
import tensorflow as tf


class Node:

    def __init__(self, parent, prob=0.0, c_puct=1.0):
        self.n_visits = 0
        self.value = 0.0
        self.children = {}
        self.parent = parent
        self.c_puct = c_puct
        self.prob = prob

        if parent == None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

    def add_child(self, action, prob):
        self.children[str(action)] = Node(
            parent=self, prob=prob, c_puct=self.c_puct)

    def is_root(self):
        return True if self.parent is None else False

    def is_leaf(self):
        return self.children == {}

    def expand(self, actions, probs):
        'Expand mcts node'
        for action, prob in zip(actions, probs):
            if str(action) not in self.children:
                self.add_child(action, prob)
        # Return random child
        action, random_child = random.choice(list(self.children.items()))
        return eval(action), random_child

    def select(self):
        action, node = max(self.children.items(),
                           key=lambda node: node[1].alpha_zero_value())
        # action is stored as str in dict, we need to convert it in list before passing it to the game
        action = eval(action)
        return action, node

    def update(self, value):
        self.n_visits += 1
        self.value += value

    def back_propagate(self, value):
        self.update(value)
        if not self.is_root():
            # parent's node refers to opponent's action
            self.parent.back_propagate(1-value)

    def get_children_nodes(self):
        return self.children.values()

    def get_children_actions(self):
        return [eval(action) for action in self.children.keys()]

    def alpha_zero_value(self):
        total_sqrt = math.sqrt(
            sum([child.n_visits for child in self.get_children_nodes()]))
        score = self._get_value_avg() + self.c_puct * self.prob * \
            total_sqrt / (1 + self.n_visits)
        return score

    def _get_value_avg(self):
        return self.value / (self.n_visits + 1)

    def _set_children_probs(self, probs):
        print(probs)
        # for child in get_children_nodes():

    def get_best_action(self):
        'Get next action based on children values. Return action,node'
        if self.is_leaf():
            raise Exception(f"Node {str(self)} do not have any child!")
        # Best action is action with minimal opponnent node value
        # action, node = min(self.children.items(),
        #                    key=lambda act_node: act_node[1].prob)
        actions_str = list(self.children.keys())
        probs = [1-node.prob for node in self.get_children_nodes()]
        action_str = np.random.choice(actions_str, p=softmax(probs))
        node = self.children[action_str]
        return eval(action_str), node

    def find_child_with_action(self, action, prob):
        action = str(action)
        # add action if not in child
        self.expand([action], [prob])

        for child_action, child_node in self.children.items():
            if child_action == action:
                return child_node

        raise Exception("No child found for action: ", action)

    def __str__(self):
        return f"Node n°{id(self)} with depth {self.depth} and value {self.value}/{self.n_visits}"


class MCTS:
    def __init__(self, goban_size, model_path=None):
        self._root = Node(parent=None, prob=1.0)
        self.goban_size = goban_size
        self.model = AlphaZeroModel(input_shape=(
            goban_size, goban_size), actions_n=goban_size**2, model_path=model_path)

    def _action_to_prob(self, action, goban):
        probs, _ = self._get_action_probs(goban)
        action_id = get_action_id(action[0], action[1], self.goban_size)
        return probs[action_id]

    def _get_action_probs(self, goban):
        legal_actions = goban.get_legal_actions()
        actions_id = [get_action_id(
            move[0], move[1], self.goban_size) for move in legal_actions]
        _, probs = self.model.predict_one(goban.get_state())
        action_probs = np.zeros(self.goban_size**2)
        _action_to_prob = {}

        for action_id, action in zip(actions_id, legal_actions):
            action_probs[action_id] = probs[action_id]
            _action_to_prob[str(action)] = probs[action_id]

        return action_probs, _action_to_prob

    def _set_children_probs(self, node, goban):
        _, _action_to_prob = self._get_action_probs(goban)
        for action, node in node.children.items():
            node.prob = _action_to_prob[action]

    def _get_value(self, state):
        value, _ = self.model.predict_one(state)
        return value

    def _get_children_values(self, node, goban):
        values = np.zeros(self.goban_size**2, dtype="float32")
        children = node.get_children_nodes()
        actions = node.get_children_actions()

        for action, child in zip(actions, children):
            action_id = get_action_id(action[0], action[1], self.goban_size)
            values[action_id] = child._get_value_avg()

        return values

    def find_child_with_action(self, node, action, goban):
        'Return: child of the given node corresponding to the action'
        prob = self._action_to_prob(action, goban)
        return node.find_child_with_action(action, prob)

    def train(self, starting_goban, n_episodes=25, epochs=10, fit_every_episode=5, batch_size=32, learning_rate=1e-2, verbose=True):
        """
        Train model

        Parameters:
                starting_goban: Goban
                n_episodes: Number of games to simulate (optional, default=25)
                epochs: Number of epochs to train the model with (optional, default=10)
                fit_every_episode: Number of episodes to wait until fitting the model (optional, default=5)
                bath_size: Number of states to fit the model with (optional, default=32)
                learning_rate: Optimizer learning rate (optional, default=1e-2)
                verbose: Print training informations (optional, default=True)
        """
        batch = []
        for episode in range(1, n_episodes+1):
            goban_copy = copy.deepcopy(starting_goban)
            node = self._root
            start_episode_batch_id = len(batch)

            while not goban_copy.is_game_over():
                state = goban_copy.get_state()
                self._set_children_probs(node, goban_copy)
                self.search_batch(10, batch_size, goban_copy, node)
                true_probs = self._get_children_values(node, goban_copy)

                batch.append([state, None, true_probs])
                action, node = node.get_best_action()
                goban_copy.play(
                    (action[0], action[1], goban_copy._next_player))

            # Set game result
            result = goban_copy.get_winner()
            for b in batch[start_episode_batch_id:]:
                b[1] = [result]

            if episode % fit_every_episode == 0:
                if verbose:
                    print("Finished episode", episode, "/", n_episodes)
                self.model.fit(batch, epochs,
                               batch_size, verbose=verbose, learning_rate=learning_rate)
                batch.clear()

    def search_batch(self, count, batch_size, goban, starting_node=None):
        """
        Proceed searchs for best next play

        Parameters:
                count: Number of searchs to perform
                batch_size: Number of monte carlo steps
                goban: Goban on which starting search
                starting_node: Node on which starting search (default=root node)
        """
        for _ in range(count):
            self.search_one_minibatch(
                batch_size, goban, starting_node)

    def search_one_minibatch(self, minibatch_size, goban, starting_node=None):
        """
        Proceed one search for best next play

        Parameters:
            minibatch_size: Number of monte carlo steps
            goban: Goban on which starting search
            starting_node: Node on which starting search (default=root node)
        """
        backprop_queue = []
        expand_queue = []
        planned = set()

        for _i_minibatch in range(minibatch_size):
            # Selections
            node = self._root if starting_node is None else starting_node
            goban_copy = copy.deepcopy(goban)

            # Find leaf
            while True:
                if node.is_leaf():
                    break
                else:
                    action, node = node.select()
                    goban_copy.play(action)

            if goban_copy.is_game_over():
                backprop_queue.append(
                    (node, goban_copy.get_winner()))
            else:
                if node not in planned:
                    planned.add(node)
                    # Simulation step replaced by a NN
                    probs, _ = self._get_action_probs(goban_copy)
                    value = self._get_value(goban_copy.get_state())

                    expand_queue.append(
                        (node, probs, value, goban_copy.get_legal_actions()))

        # Expansion :
        if expand_queue:
            for node, probs, value, actions in expand_queue:
                node.expand(actions, probs)
                backprop_queue.append((node, value))

        # Backpropagation
        for node, value in backprop_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            node.back_propagate(cur_value)


class AlphaZeroPlayer(PlayerInterface):
    def __init__(self, color, goban_size=7, model_path=None):
        super().__init__(color, goban_size)
        self.mcts = MCTS(goban_size, model_path)
        self._current_node = self.mcts._root

    def _update_current_node_with_action(self, action):
        # new_current_node = self._current_node.find_child_with_action(action)
        new_current_node = self.mcts.find_child_with_action(
            self._current_node, action, self.goban)
        self._update_current_node(new_current_node)

    def _update_current_node(self, node):
        self._current_node = node

    def _play_mcts(self):
        start = time.time()
        self.mcts.search_batch(
            10, 16, self.goban, starting_node=self._current_node)
        print("ZERO took", time.time()-start)

    def getPlayerName(self):
        return f"AlphaZero player {self.color}"

    def getPlayerMove(self):
        'Compute best move accoding to the player policy'
        self._play_mcts()
        action, node = self._current_node.get_best_action()
        assert action[2] == self.color

        self._update_current_node(node)
        self.goban.play(action)
        return action

    def playOpponentMove(self, x, y):
        'Play opponent move on player board'
        assert(self.goban.is_valid_move((x, y, self._opponent)))
        self.goban.play((x, y, self._opponent))
        action = (x, y, self._opponent)
        self._update_current_node_with_action(action)

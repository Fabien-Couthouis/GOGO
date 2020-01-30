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
from goban import Goban
import math
import time
from players.alphazeroModel import AlphaZeroModel
from players.playerInterface import PlayerInterface


class Node:

    def __init__(self, parent, mcts, prob=1.0, c_puct=1.0):
        self.n_visits = 0
        self.value = 0
        self.children = {}
        self.parent = parent
        self.prob = prob
        self.mcts = mcts
        self.c_puct = c_puct

        if parent == None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

    def add_child(self, action, prob):
        self.children[str(action)] = Node(
            parent=self, mcts=self.mcts, prob=prob, c_puct=self.c_puct)

    def is_root(self):
        return True if self.parent is None else False

    def is_leaf(self):
        return self.children == {}

    def expand(self, actions, probs):
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

    def alpha_zero_value(self):
        total_sqrt = math.sqrt(
            sum([child.n_visits for child in self.get_children_nodes()]))
        # if self.is_root():
        #     # The root node has an extra noise added to the probabilities to improve the exploration
        #     noises = np.random.dirichlet(
        #         (0.03 * self.mcts.goban_size, 0.03 * self.mcts.goban_size))
        #     prob = [0.75 * prob + 0.25 * noise for prob,
        #              noise in zip(probs, noises)]
        score = self.get_value_avg() + self.c_puct * self.prob * \
            total_sqrt / (1 + self.n_visits)
        return score

    def get_value_avg(self):
        return self.value / (self.n_visits + 1)

    def get_best_action(self):
        'Get next action based on children values. Return action,node'
        if self.is_leaf():
            raise Exception(f"Node {str(self)} do not have any child")
        # Best action is action with minimal opponnent node value
        action, node = min(self.children.items(),
                           key=lambda act_node: act_node[1].value)
        return eval(action), node

    def find_child_with_action(self, action):
        action = str(action)
        # add action if not in child
        # self.expand([action])

        for child in self.children.items():
            child_action, child_node = child
            if child_action == action:
                return child_node

        raise Exception("No child found for action: ", action)

    def __str__(self):
        return f"Node n°{id(self)} with depth {self.depth} and value {self.value}/{self.n_visits}"


class MCTS:
    def __init__(self, goban_size):
        self._root = Node(parent=None, mcts=self)
        self.goban_size = goban_size
        self._n_simulations = 0
        self.model = AlphaZeroModel(input_shape=(
            goban_size, goban_size), actions_n=goban_size**2)

    def get_n_simulations(self):
        return self._n_simulations

    def train(self, starting_goban, n_episodes=1000, batch_size=16, verbose=True):
        batch = []
        for episode in range(1, n_episodes+1):
            goban_copy = copy.deepcopy(starting_goban)
            node = self._root

            while not goban_copy.is_game_over():
                self.search_batch(10, batch_size, goban_copy, node)
                action, node = node.get_best_action()
                batch.append(
                    (goban_copy.get_state(), action, node.get_value_avg()))

            if episode % batch_size == 0:
                self.model.fit(batch)
                batch.clear()

            if verbose and episode % (n_episodes * 0.05) == 0:
                print("Finished episode", episode, "/", n_episodes)

    def search_batch(self, count, batch_size, goban, starting_node=None):
        for _ in range(count):
            self.search_one_minibatch(
                batch_size, goban, starting_node)

    def search_one_minibatch(self, minibatch_size, goban, starting_node=None):
        backprop_queue = []
        expand_states = []
        expand_queue = []
        planned = set()

        for _i_minibatch in range(minibatch_size):
            # Selections
            # Start from root R and select successive child nodes until a leaf node L is reached.
            node = self._root if starting_node is None else starting_node
            goban_copy = copy.deepcopy(goban)
            states, actions = [], []
            while not node.is_leaf():
                states.append(goban_copy.get_state())
                actions.append(goban_copy.get_legal_actions())
                action, node = node.select_alphazero()
                goban_copy.play(action)

            if goban_copy.is_game_over():
                backprop_queue.append(
                    (goban_copy.get_winner(), states, actions))
            else:
                if node not in planned:
                    planned.add(node)
                    expand_states.append(states)
                    expand_queue.append(
                        (goban_copy.get_state(), states, actions))

            # Expansion :
            if expand_queue:
                # Simulation step replaced by NN
                # probs, values = self.model.predict(
                #     np.expand_dims(goban_copy.get_state(), axis=0))
                state = goban_copy.get_state()
                print(state.shape)
                probs, values = self.model.predict(state)

                # create the nodes
                action, node = node.expand(goban_copy.get_legal_actions())
                goban_copy.play(action)
                for (node, states, actions), prob, value in zip(expand_queue, probs, values):
                    action, node = node.expand(actions, prob)
                    backprop_queue.append((value, states, actions))

            # Backpropagation
            for value, states, actions in backprop_queue:
                # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
                cur_value = -value
                node.back_propagate(cur_value)

    def set_root(self, node):
        self._root = node

    def save(self, path="mcts.pickle"):
        "Save mcts object using joblib"
        import joblib
        joblib.dump(self, path, compress=4)


class AlphaZeroPlayer(PlayerInterface):
    def __init__(self, color, goban_size=7, mcts=None):
        super().__init__(color, goban_size)
        self.mcts = mcts if mcts is not None else MCTS(goban_size)
        self._current_node = self.mcts._root

    def _update_current_node_with_action(self, action):
        new_current_node = self._current_node.find_child_with_action(action)
        self._update_current_node(new_current_node)

    def _update_current_node(self, node):
        self._current_node = node

    def _play_mcts(self):
        start = time.time()
        self.mcts.search_batch(
            10, 12, self.goban, starting_node=self._current_node)
        print("ZERO took", time.time()-start)

    def getPlayerName(self):
        return f"AlphaZero player {self.color}"

    def getPlayerMove(self):
        self._play_mcts()
        action, node = self._current_node.get_best_action()
        assert action[2] == self.color

        self._update_current_node(node)
        self.goban.play(action)
        return action

    def playOpponentMove(self, x, y):
        assert(self.goban.is_valid_move((x, y, self._opponent)))
        self.goban.play((x, y, self._opponent))
        action = (x, y, self._opponent)
        self._update_current_node_with_action(action)

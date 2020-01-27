import random
from players.playerInterface import PlayerInterface
from operator import itemgetter
import random
import copy
import numpy as np
from goban import Goban
import math
import time
from models import AlphaZeroModel
from players.playerInterface import PlayerInterface


def alphazero_policy(goban, model):
    'NN policy'
    legal_actions = goban.get_legal_actions()
    _prior, value = model.predict(goban.get_board())
    return value


def get_prior(goban, model):
    prior, _value = model.predict(goban.get_board())
    return prior


def rollout_policy(goban):
    'Rollout randomly'
    legal_actions = goban.get_legal_actions()
    action_probs = np.random.rand(len(legal_actions))
    return zip(legal_actions, action_probs)


class Node:

    def __init__(self, parent, mcts):
        self.n_visits = 0
        self.value = 0
        self.children = {}
        self.parent = parent
        self.mcts = mcts

        if parent == None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

    def add_child(self, action):
        self.children[str(action)] = Node(parent=self, mcts=self.mcts)

    def is_root(self):
        return True if self.parent is None else False

    def is_leaf(self):
        return self.children == {}

    def expand(self, actions):
        for action in actions:
            if str(action) not in self.children:
                self.add_child(action)
        # Return random child
        action, random_child = random.choice(list(self.children.items()))
        return eval(action), random_child

    def select(self):
        action, node = max(self.children.items(),
                           key=lambda node: node[1].ucb1())
        # action is stored as str in dict, we need to convert it in list before passing it to the game
        action = eval(action)
        return action, node

    def select_alphazero(self, prior):
        action, node = max(self.children.items(),
                           key=lambda node: node[1].alpha_zero_value(prior))
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

    # def alpha_zero_value(self, prior, gamma=0.5):
    #     return gamma * self.ucb1() + (1-gamma) * prior

    def ucb1(self):
        return self.value / (self.n_visits+1) + math.sqrt(2 * math.log(self.mcts.get_n_simulations()) / (self.n_visits+1))

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
        self.expand([action])

        for child in self.children.items():
            child_action, child_node = child
            if child_action == action:
                return child_node

        raise Exception("No child found for action: ", action)

    def __str__(self):
        return f"Node n°{id(self)} with depth {self.depth} and value {self.value}/{self.n_visits}"


class Experience:
    def __init__(self):
        self.states = []
        self.actions = []
        self.reward = None

    def add_play(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    def get_states(self):
        return np.array(self.states)

    def get_rewards(self):
        return np.array([self.reward]*len(self.states))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.reward = None


class MCTS:
    def __init__(self, goban_size):
        self._root = Node(parent=None, mcts=self)
        self._n_simulations = 0
        self.model = AlphaZeroModel(input_shape=(
            goban_size, goban_size), hidden_size=64)

    def get_n_simulations(self):
        return self._n_simulations

    def update_model(self, experiences, batch_size):
        for i in range(batch_size):
            experience = random.sample(experiences)

            with tf.GradientTape() as tape:
                priors, values = self.model.predict(experience.get_states())
                value_loss = tf.keras.losses.mean_squared_error(
                    experience.get_rewards(), values)
                prior_loss = tf.keras.losses.categorical_crossentropy(
                    experience.priors, priors)

            gradients = tape.gradient(
                total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

    def train(self, starting_goban, n_episodes=1000, batch_size=16, starting_node=None, verbose=True):
        experiences = []
        current_batch = 0
        for episode in range(1, n_episodes+1):
            current_batch += 1
            goban_copy = copy.deepcopy(goban)
            experience = Experience()

            while not goban_copy.is_game_over():
                action = self.mcts_one_iteraction(
                    starting_goban, starting_node)
                self.experience.add_play(goban_copy.get_board(), action)

            experience.add_reward(goban_copy.get_winner())
            experiences.append(experience)

            if current_batch == 16:
                self.update_model(experiences)
                current_batch = 0

            if verbose and episode % (n_episodes * 0.05) == 0:
                print("Finished episode", episode, "/", n_episodes)

    def mcts_one_iteraction(self, goban, starting_node=None):
        """
        Performs a round of Monte Carlo: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
        """

        # Selection
        # Start from root R and select successive child nodes until a leaf node L is reached.
        node = self._root if starting_node is None else starting_node
        while not node.is_leaf():
            prior = get_prior(self.model)
            # Take ucb-directed action
            action, node = node.select_alphazero(prior)
            goban_copy.play(action)

        # Expansion :
        # Unless L ends the game decisively (e.g. win/loss/draw) for either player
        # create one (or more) child nodes and choose node C from one of them.
        if not goban_copy.is_game_over():
            action, node = node.expand(goban_copy.get_legal_actions())
            goban_copy.play(action)

        # Simulation
        # Complete one random rollout from node C
        # value = self.simulate(goban_copy)  # TODO: neural network estimation
        value = alphazero_policy(goban, self.model)

        # Backpropagation
        # Use the result of the playout to update information in the nodes above
        node.back_propagate(value)
        return action

    def set_root(self, node):
        self._root = node

    def simulate(self, goban, limit=1000):
        """
        Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, 0 if the opponent wins,
        and 0.5 if it is a tie.
        """
        current_player = goban.get_next_player()
        while not goban.is_game_over():
            action_probs = rollout_policy(goban)
            next_action = max(action_probs, key=itemgetter(1))[0]
            goban.play(next_action)

        winner = goban.get_winner()
        self._n_simulations += 1
        # tie
        if winner == 0:
            return 0.5
        # current player wins
        elif winner == current_player:
            return 1
        # current player loses
        else:
            return 0

    def save(self, path="mcts.pickle"):
        "Save mcts object using joblib"
        import joblib
        joblib.dump(self, path, compress=4)


class MCTSPlayer(PlayerInterface):
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
        # TODO: take time into account (see Reversi Timer on Github)
        for i in range(10):
            self.mcts.mcts_one_iteraction(
                self.goban, starting_node=self._current_node)
        print("MCTS took", time.time()-start)

    def getPlayerName(self):
        return f"MCTS player {self.color}"

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

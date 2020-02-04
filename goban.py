#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre
#############################################################
import numpy as np
from utils.unionFind import UF
from utils.Trans import Zob


def get_action_id(x, y, board_size):
    return x + board_size*y


class Stone:
    """
    Plays abstraction and id calculation for union-find
    """

    def __init__(self, x, y, board_size, color=-1):
        self.x = x
        self.y = y
        self.color = color
        self.id = get_action_id(self.x, self.y, board_size)

    def __str__(self):
        return f"Stone with color: {self.color} and id: id {self.id} placed in position ({self.x},{self.y})"


class Goban:
    """
    Go simulator (simplified Tromp-Taylor rules: https://senseis.xmp.net/?TrompTaylorRules).

    Usefull methods: get_board(), get_legal_actions(), get_score(), get_winner(), is_game_over(), play(action)

    """
    _EMPTY = -1
    _WHITE = 0
    _BLACK = 1

    def __init__(self, board_size=7):
        self.board_size = board_size
        self.n_turns = 0

        self._size = board_size
        self._white_uf = UF(self._size*self._size)
        self._black_uf = UF(self._size*self._size)

        self._white_score, self._black_score = 0, 0
        self._consecutive_turns_passed = 0
        self._next_player = self._BLACK
        self._last_move = None

        # Generate _board
        self._board = []
        for _ in range(self._size):
            self._board.append([self._EMPTY] * self._size)
        self.zob = Zob(self)  # Zob transtable
        self._empty_pos = [(x, y) for x in range(self._size)
                           for y in range(self._size)]

    def _is_pos_valid(self, x, y):
        'True if the given position stays whithin the board limits, False otherwise.'
        return False if (x < 0 or x >= self._size or y < 0 or y >= self._size) else True

    def _get_neighbors(self, stone):
        'Get all the stones adjacent to the given one.'
        neighbors = []
        neighbors_pos = [(stone.x-1, stone.y), (stone.x+1, stone.y),
                         (stone.x, stone.y-1), (stone.x, stone.y+1)]
        for x_n, y_n in neighbors_pos:
            if self._is_pos_valid(x_n, y_n):
                color = self._get_color_from_pos(x_n, y_n)
                neighbors.append(Stone(x_n, y_n, self.board_size, color))
        return neighbors

    def _get_uf(self, color):
        if color == Goban._WHITE:
            return self._white_uf
        else:
            return self._black_uf

    def _make_union_with_neighbors(self, stone):
        'Make union with neighbors in the same color in the union-find corresponding to the stone color.'
        uf = self._get_uf(stone.color)
        neighbors = self._get_neighbors(stone)
        for neighbor in neighbors:
            if stone.color == neighbor.color:
                uf.union(stone.id, neighbor.id)

    def _get_color_from_pos(self, x, y):
        return self._board[x][y]

    def _get_stone_from_id(self, stone_id):
        'Generate Stone object from id'
        x = stone_id % self._size
        y = stone_id // self._size
        color = self._get_color_from_pos(x, y)
        return Stone(x, y, self.board_size, color)

    def _put_stone(self, stone):
        'Put stone on the board, update chains and update scores.'
        self._board[stone.x][stone.y] = stone.color
        # Store the board in zob transtable
        self.zob.store()
        self._make_union_with_neighbors(stone)
        self._update_adjacent_opponent_chains(stone)

        # Update score
        if stone.color == Goban._WHITE:
            self._white_score += 1
        else:
            self._black_score += 1

    def _get_chain(self, stone):
        'Get the chain in which the stone belongs.'
        # Stone not put on _board
        if self._get_color_from_pos(stone.x, stone.y) == Goban._EMPTY:
            raise Exception(
                "Stone is not on the board, so its chain does not exist!")

        uf = self._get_uf(stone.color)
        set_id = uf.find(stone.id)
        chain = uf.get_chain(set_id)
        return chain

    def _get_chain_liberties(self, chain):
        'Get number of liberties for the chain.'
        if chain is None:
            raise Exception("Chain do not exists")

        liberties = 0
        # Iterate over all stones in the chain
        for stone_id in chain:
            stone = self._get_stone_from_id(stone_id)
            neighbors = self._get_neighbors(stone)
            # Increment liberties for each empty neighbor
            for neighbor in neighbors:
                if neighbor.color == Goban._EMPTY:
                    liberties += 1

        return liberties

    def _update_adjacent_opponent_chains(self, stone):
        'Check if adjacent chains from opponent has liberties = 0 and delete it if so.'
        neighbors = self._get_neighbors(stone)
        opponent_color = self.invert_color(stone.color)

        for neighbor in neighbors:
            # Need to update color as stone may be deleted in the loop
            neighbor.color = self._get_color_from_pos(neighbor.x, neighbor.y)
            if neighbor.color == opponent_color:
                chain = self._get_chain(neighbor)
                liberties = self._get_chain_liberties(chain)
                if liberties == 0:
                    self._delete_chain(chain)

    def _delete_chain(self, chain):
        'Delete all stones from the chain.'
        def delete_from_board(stone):
            self._board[stone.x][stone.y] = Goban._EMPTY
            # Append removed stone to lastmove
            self._last_move[1].append(stone)
            # And to empty stone list
            if not (stone.x, stone.y) in self._empty_pos:
                self._empty_pos.append((stone.x, stone.y))

        for stone_id in chain:
            stone = self._get_stone_from_id(stone_id)
            delete_from_board(stone)

            # Update score
            if stone.color == Goban._WHITE:
                self._white_score -= 1
            else:
                self._black_score -= 1

    def get_next_player(self):
        return self._next_player

    def get_last_move(self):
        return self._last_move

    def get_score(self):
        'Return: _black_score, _white_score.'
        return self._black_score, self._white_score

    def get_board_size(self):
        return self._size

    def get_board(self):
        return self._board

    def get_state(self):
        'Get board state for current player'
        if self._next_player == self._BLACK:
            state = self.get_board()
        else:
            reversed_board = [
                [self.invert_color(stone) for stone in row] for row in self.get_board()]
            state = reversed_board
        state = np.array(state)
        state = np.expand_dims(state, axis=-1)
        return state

    def get_winner(self):
        'Return none if no winner, else return the color of the winner (_EMPTY color if tie)'
        if self.is_game_over():
            b, w = self.get_score()
            if w > b:
                return self._WHITE
            elif b > w:
                return self._BLACK
            else:
                return self._EMPTY  # tie
        else:
            return None

    def get_legal_actions(self):
        'Get a list of legal actions tuples (x, y, color)'
        # TODO: check this function, add illegal moves
        legal_moves = []
        for x, y in self._empty_pos:
            if not self.zob.is_already_played((x, y, self._next_player)):
                legal_moves.append((x, y, self._next_player))

        if len(legal_moves) == 0:
            # We shall pass
            legal_moves = [(-1, -1, self._next_player)]

        return legal_moves

    def is_valid_move(self, action):
        'Return True if the action tuple (x, y, color) is valid on the board, else False'
        x, y, color = action
        if not self._is_pos_valid(x, y):
            # pass if pass :)
            if x == -1 and y == -1:
                pass
            else:
                raise Exception("Move not valid (out of board)")

        elif self._get_color_from_pos(x, y) != Goban._EMPTY:
            raise Exception("Move not valid (stone already here)")

        elif color != self._next_player:
            raise Exception("This is not your turn, player", color)

        return True

    def play(self, action):
        'Append the action tuple (x, y, color) to the Goban'

        x, y, color = action
        assert self.is_valid_move(action)

        # Pass
        if x == -1 and y == -1:
            self._consecutive_turns_passed += 1

        else:
            stone = Stone(x, y, self.board_size, color)
            # Add stone to last move and put it on the board
            self._last_move = (stone, [])
            self._last_move[1].append(stone)
            self._put_stone(stone)

            self._consecutive_turns_passed = 0
            self._empty_pos.remove((stone.x, stone.y))

        self._next_player = self.invert_color(self._next_player)
        self.n_turns += 1

    def is_game_over(self):
        if self.n_turns > self._size**2:
            return True
        elif self._consecutive_turns_passed == 2:
            return True
        else:
            return False

    def invert_color(self, color):
        'Return black if given color is white and vice versa'
        if color not in [Goban._BLACK, Goban._WHITE]:
            return Goban._EMPTY
        return Goban._BLACK if color == Goban._WHITE else Goban._WHITE

    def __str__(self):
        def color_to_str(color):
            if color == Goban._WHITE:
                return 'O'
            elif color == Goban._BLACK:
                return 'X'
            else:
                return '.'

        to_return = ""
        for j in range(self._size):
            for i in range(self._size):
                color = self._get_color_from_pos(i, j)
                to_return += color_to_str(color)
            to_return += "\n"

        return to_return

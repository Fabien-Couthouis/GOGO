#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre

from UF import UF
from Trans import Zob
_BOARD_SIZE = 9


class Stone:
    def __init__(self, x, y, color=-1):
        self.x = x
        self.y = y
        self.color = color
        self.id = self.x + _BOARD_SIZE*self.y

    def __str__(self):
        return f"Stone with color: {self.color} and id: id {self.id} placed in position ({self.x},{self.y})"


# TODO: fin du game ?
class Goban:
    _EMPTY = -1
    _WHITE = 0
    _BLACK = 1

    def __init__(self):
        print("je sais pas encore how to play")
        self.size = _BOARD_SIZE
        self.white_uf = UF(self.size*self.size)
        self.black_uf = UF(self.size*self.size)
        self.white_score = 0
        self.black_score = 0
        self.n_turns = 0
        self.consecutive_turns_passed = 0
        self.last_move = None
        # Generate board
        self.board = []
        for _ in range(self.size):
            self.board.append([self._EMPTY] * self.size)
        self.zob = Zob(self)

    def _is_pos_valid(self, x, y):
        return False if (x < 0 or x >= self.size or y < 0 or y >= self.size) else True

    def _get_neighbors(self, stone):
        'Get all the stones adjacent to the given one.'
        neighbors = []
        neighbors_pos = [(stone.x-1, stone.y), (stone.x+1, stone.y),
                         (stone.x, stone.y-1), (stone.x, stone.y+1)]
        for x_n, y_n in neighbors_pos:
            if self._is_pos_valid(x_n, y_n):
                color = self._get_color_from_pos(x_n, y_n)
                neighbors.append(Stone(x_n, y_n, color))
        return neighbors

    def _get_uf(self, color):
        if color == Goban._WHITE:
            return self.white_uf
        else:
            return self.black_uf

    def _make_union_with_neighbors(self, stone):
        'Make union with neighbors in the same color in the union-find corresponding to the stone color.'
        uf = self._get_uf(stone.color)
        neighbors = self._get_neighbors(stone)
        for neighbor in neighbors:
            if stone.color == neighbor.color:
                uf.union(stone.id, neighbor.id)

    def _get_color_from_pos(self, x, y):
        return self.board[x][y]

    def _get_stone_from_id(self, stone_id):
        'Generate Stone object from id'
        x = stone_id % self.size
        y = stone_id // self.size
        color = self._get_color_from_pos(x, y)
        return Stone(x, y, color)

    def _put_stone(self, stone):
        print("PUT STONE:", stone)
        self.board[stone.x][stone.y] = stone.color
        self._make_union_with_neighbors(stone)
        self._update_adjacent_opponent_chains(stone)

        # Update score
        if stone.color == Goban._WHITE:
            self.white_score += 1
        else:
            self.black_score += 1

    def _get_chain(self, stone):
        'Get the chain in which the stone belongs. Return None if the stone is not put on the board.'
        # Stone not put on board
        if self._get_color_from_pos(stone.x, stone.y) == Goban._EMPTY:
            return None

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
            if neighbor.color == opponent_color:
                chain = self._get_chain(neighbor)
                liberties = self._get_chain_liberties(chain)
                if liberties == 0:
                    self._delete_chain(chain)

    def _delete_chain(self, chain):
        'Delete all stones from the chain.'
        def delete_from_board(stone):
            self.board[stone.x][stone.y] = Goban._EMPTY
            # Append removed stone from the board
            self.last_move[1].append(stone)

        for stone_id in chain:
            stone = self._get_stone_from_id(stone_id)
            delete_from_board(stone)
            # Update score
            if stone.color == Goban._WHITE:
                self.white_score -= 1
            else:
                self.black_score -= 1

    def play(self, x, y, color):
        'Append the play to the Goban'
        if not self._is_pos_valid(x, y):
            raise Exception("Move not valid (out of board)")
        if self._get_color_from_pos(x, y) != Goban._EMPTY:
            raise Exception("Move not valid (stone already here)")

        stone = Stone(x, y, color)
        # Add stone to last move
        self.last_move = (stone, [])
        self.last_move[1].append(stone)
        print(self.last_move)
        self._put_stone(stone)
        self.n_turns += 1

        if x == -1 and y == -1:
            self.consecutive_turns_passed += 1
        else:
            self.consecutive_turns_passed = 0
        self.zob.store()

    def get_last_move(self):
        return self.last_move

    def is_game_ended(self):
        if self.n_turns > self.size**2:
            return True
        elif self.consecutive_turns_passed == 2:
            return True
        elif self.zob.is_a_board_multiple_times():
            return True
        else:
            return False

    def get_score(self):
        'Return: black_score, white_score.'
        return self.black_score, self.white_score

    def invert_color(self, color):
        if color not in [Goban._BLACK, Goban._WHITE]:
            raise Exception("Cannot invert color", color)
        return Goban._BLACK if color == Goban._WHITE else Goban._BLACK

    def get_board_size(self):
        return self.size

    def get_board(self):
        return self.board

    def __str__(self):
        def color_to_str(color):
            if color == Goban._WHITE:
                return 'O'
            elif color == Goban._BLACK:
                return 'X'
            else:
                return '.'

        to_return = ""
        for j in range(self.size):
            for i in range(self.size):
                color = self._get_color_from_pos(i, j)
                to_return += color_to_str(color)
            to_return += "\n"

        return to_return


if __name__ == "__main__":
    # LES TESTS (ils sont supers !)
    goban = Goban()
    goban.play(1, 3, Goban._WHITE)
    goban.play(1, 5, Goban._WHITE)
    goban.play(1, 4, Goban._BLACK)
    goban.play(0, 4, Goban._WHITE)
    goban.play(2, 4, Goban._WHITE)

    print(goban)
    chain = goban._get_chain(Stone(1, 5, Goban._WHITE))
    print(goban._get_chain_liberties(chain))
    print(goban.get_last_move())
    print(goban.zob.data)

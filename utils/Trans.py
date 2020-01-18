import random
from copy import deepcopy
# Implementation d'une table de transposition améliorée en utilisant le hashage de Zobrist


class Zob:
    def __init__(self, b):
        self.b = b
        (self._BLACK, self._WHITE, self._EMPTY) = (
            self.b._BLACK, self.b._WHITE, self.b._EMPTY)
        self.zobTable = [[[random.randint(1, 2**10-1) for i in range(2)] for j in range(
            self.b.get_board_size())] for k in range(self.b.get_board_size())]
        self.hash = self.compute_hash()
        self.data = []

    # Retourne l'index dans la table de Zobrist correspondant à chaque type de pièce
    def indexing(self, piece):
        if piece == self._BLACK:
            return 0
        else:
            return 1

    # Calcul du hachage de l'état initial du plateau de jeu
    def compute_hash(self, goban_board=None):
        h = 0
        board = self.b.get_board() if goban_board is None else goban_board
        for i in range(self.b.get_board_size()):
            for j in range(self.b.get_board_size()):
                if board[i][j] != self._EMPTY:
                    stone_color = board[i][j]
                    h ^= self.zobTable[i][j][self.indexing(stone_color)]
        return h

    # Mise à jour du hachage suivant le dernier coup joué à l'aide d'une succession de XOR (plus rapide que de recalculer le hachage du plateau de jeu entier)
    def update_hash(self, stone, changed_stones):
        self.hash ^= self.zobTable[stone.x][stone.y][self.indexing(
            stone.color)]
        for s in changed_stones:
            self.hash ^= self.zobTable[s.x][s.y][self.indexing(
                self.b.invert_color(s.color))]

            self.hash ^= self.zobTable[s.xf][s.yf][self.indexing(s.color)]

    def store(self):
        """Stockage de l'état de jeu dans la table de transposition via son hachage"""
        self.data.append(self.compute_hash())

    def is_already_played(self, action):
        """Return True if the play has already be seen"""
        x, y, color = action
        goban_board_copy = deepcopy(self.b.get_board())
        goban_board_copy[x][y] = color
        play_hash = self.compute_hash(goban_board_copy)
        return play_hash in self.data

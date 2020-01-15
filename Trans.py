import random

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
    def compute_hash(self):
        h = 0
        board = self.b.get_board()
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

    # On regarde si l'état du plateau correspond à un état déjà enregistré dans la table (à l'aide de son hachage)
    # et on le renvoie le cas échéant
    def lookup(self):
        (played_stone, changed_stones) = self.b.get_last_move()
        if played_stone is not None:
            self.update_hash(played_stone, changed_stones)
        return self.data.get(self.hash, None)

    def store(self):
        """Stockage de l'état de jeu dans la table de transposition via son hachage"""
        self.data.append(self.compute_hash())

    def is_a_board_multiple_times(self):
        """Return True if a board has been seen multiple times"""
        return len(self.data) != len(list(set(self.data))

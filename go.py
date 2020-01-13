from python_algorithms.basic.union_find import UF
import numpy as np


def encode_stone_pos(x, y):
    return self.size**2*x + y


def unencode_stone_pos(xy):
    x = xy//self.size**2
    y = xy % self.size**2
    return (x, y)


def get_neighbors((x, y)):
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]


class Stone:
    def __init__(self, color, x, y):
        self.color = color
        self.x = x
        self.y = y


class Chain:
    def __init__(self, color, chain_id, size):
        self.color = color
        self.chain_id = chain_id
        self.liberties = 0


class Goban:
    _EMPTY = -1
    _WHITE = 0
    _BLACK = 1

    def __init__(self, size):
        print("je sais pas encore how to play")
        self.size = size
        self.whites = UF(self.size*self.size)
        self.blacks = UF(self.size*self.size)
        self.white_chains = np.array(self.size*self.size)
        self.black_chains = np.array(self.size*self.size)

    def get_uf(self, color):
        if color == _WHITE:
            return self.whites
        else:
            return self.blacks

    def get_chains(self, color):
        if color == _WHITE:
            return self.white_chains
        else:
            return self.black_chains

    def delete_chain(self, old_id, color):
        chain = self.get_chains(color)
        chain = np.delete(chain, old_id)

    def put_stone(self, stone):
        neighbors = [(stone.x-1, stone.y), (stone.x+1, stone.y),
                     (stone.x, stone.y-1), (stone.x, stone.y+1)]
        uf = get_uf(stone.color)
        encoded_stone = encode_stone_pos((stone.x, stone.y))

        chain = Chain(stone.color, encoded_stone, 1)
        neighbors_old_ids = []

        for position in neighbors:
            encoded_neighbor = encode_stone_pos(position)
            neighbors_old_ids.append(uf.find(encoded_neighbor))
            uf.union(encoded_stone, encoded_neighbor)

        for i, position in enumerate(neighbors):
            encoded_neighbor = encode_stone_pos(position)
            new_id = uf.find(encoded_neighbor)
            if new_id != neighbors_old_ids[i]:
                self.delete_chain(
                    neighbors_old_ids[i], self.get_color_from_pos(position))

        chain_id = uf.find(encoded_stone)

        if chain_id == encoded_stone:
            chains = self.get_chains(stone.color)
            chains[encoded_stone] = chain
            # TODO: DOC, tests

    def get_color_from_pos(self, pos):
        chain_id = encode_stone_pos(pos)

        if self.whites._rank[chain_id] != 0:
            return _WHITE
        elif self.blacks._rank[chain_id] != 0:
            return _BLACK
        else:
            return _EMPTY

    def update_liberties(self, chain):
        l = 0
        uf = self.get_uf(chain.color)

        for s in uf._id:
            if s == chain.chain_id:
                stone = unencode_stone_pos(s)
                neighbors = get_neighbors(stone)

                for n in neighbors:
                    c = self.get_color_from_pos(n)
                    if c == _EMPTY:
                        l += 1

        chain.liberties = l

        # # PEUT NE PAS MARCHER ATTENTION (mais osef parce que ça sert pas trop)
        # def find_stone(self,position,color):
        #     stone_list=get_stone_list(stone.color)
        #     for uf in stone_list:
        #         encoded_stone = self.encode_stone_pos(position)
        #         chain_id = uf.find(encoded_stone)
        #
        #     return None

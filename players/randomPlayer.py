#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre
#############################################################

import random
from players.playerInterface import PlayerInterface


class RandomPlayer(PlayerInterface):
    def __init__(self, color, board_size=7):
        super().__init__(color, board_size)

    def getPlayerName(self):
        return f"Random player {self.color}"

    def getPlayerMove(self):
        legal_moves = self.goban.get_legal_actions()
        action = random.choice(legal_moves)
        assert action[2] == self.color
        self.goban.play(action)
        return action

    def playOpponentMove(self, x, y):
        assert(self.goban.is_valid_move((x, y, self._opponent)))
        self.goban.play((x, y, self._opponent))

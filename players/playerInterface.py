#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre
#############################################################

from goban import Goban


class PlayerInterface():
    def __init__(self, color, board_size):
        self.goban = Goban(board_size)
        self.color = color
        self._opponent = self.goban.invert_color(self.color)

    def __str__(self):
        return f"Player: {self.getPlayerName()}"

    def getPlayerName(self):
        """
        Returns your player name, as to be displayed during the game.
        """
        return NotImplementedError

    def getPlayerMove(self):
        """
        Returns your move. The move must be a couple of two integers,
        which are the coordinates of where you want to put your piece on the board.
        Coordinates are the coordinates given by the Reversy.py methods (e.g. validMove(board, x, y)
        must be true of you play '(x,y)') You can also answer (-1,-1) as "pass".
        Note: the referee will nevercall your function if the game is over
        """
        return (-1, -1)

    def playOpponentMove(self, x, y):
        """Inform you that the oponent has played this move. You must play it with no search 
        (just update your local variables to take it into account)
        """
        pass

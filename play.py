#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre
#############################################################

import time
from players.randomPlayer import RandomPlayer
from players.mctsPlayer import MCTSPlayer
from players.alphazeroPlayer import AlphaZeroPlayer

from goban import Goban


def play(player1, player2, goban, verbose=True):
    # Black begins
    next_player, other_player = (
        player1, player2) if player1.color == goban._BLACK else (player2, player1)
    nbmoves = 1

    # total real time for each player
    totalTime = {p.getPlayerName(): 0 for p in [next_player, other_player]}
    if verbose:
        print(next_player, other_player)

    while not goban.is_game_over():
        if verbose:
            print("Referee Board:")
            print(goban)
            print("Before move", nbmoves)
        # print("Legal Moves: ", goban.get_legal_actions())

        nbmoves += 1

        currentTime = time.time()
        x, y, color = next_player.getPlayerMove()
        totalTime[next_player.getPlayerName()] += time.time() - currentTime
        if verbose:
            print(next_player.color,
                  next_player.getPlayerName(), "plays " + str((x, y)))
        if not goban.is_valid_move((x, y, color)):
            print(other_player, next_player, next_player.color)
            print("Problem: illegal move")
            break
        goban.play((x, y, next_player.color))
        other_player.playOpponentMove(x, y)

        # Invert players
        next_player, other_player = other_player, next_player

    if verbose:
        print(goban)
        print("The game is over")
    nbblacks, nbwhites = goban.get_score()
    winner = goban.get_winner()
    if verbose:
        print("Time:", totalTime)
        print("Winner: ", end="")
        if winner == goban._WHITE:
            print("WHITE")
        elif winner == goban._BLACK:
            print("BLACK")
        else:
            print("DEUCE")
        print("Final is: ", nbwhites, "whites and ", nbblacks, "blacks")
    return winner


if __name__ == "__main__":
    goban = Goban()
    model_path = "saves/test1/best/model"  # not well trained
    player1, player2 = AlphaZeroPlayer(
        goban._WHITE, goban.get_board_size(), model_path), MCTSPlayer(goban._BLACK, goban.get_board_size())
    play(player1, player2, goban)

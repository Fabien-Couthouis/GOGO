#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre
#############################################################

import os
import shutil
import time
import argparse
import sys
import random
import tensorflow as tf
from play import play
from goban import Goban
from players.alphazeroPlayer import MCTS, AlphaZeroPlayer

N_EPISODES = 30
FIT_EVERY_EPISODE = 3
MCTS_SEARCHES = 25
LEARNING_RATE = 0.01
BATCH_SIZE = 16
TRAIN_ROUNDS = 20  # more would be better
N_EPOCHS = 5

BEST_NET_WIN_RATIO = 0.5

EVALUATE_EVERY_ROUND = 5
EVALUATION_ROUNDS = 10
GOBAN_SIZE = 7


def evaluate(model1_path, model2_path, rounds, device="cpu"):
    win1, win2 = 0, 0
    # Turn verbosity off
    sys.stdout = open(os.devnull, 'w')

    for _round in range(rounds):
        goban = Goban(GOBAN_SIZE)
        # Random color asignment
        color1 = random.choice([Goban._BLACK, Goban._WHITE])
        color2 = goban.invert_color(color1)
        p1 = AlphaZeroPlayer(color1, GOBAN_SIZE, model1_path)
        p2 = AlphaZeroPlayer(color2, GOBAN_SIZE, model2_path)

        winner = play(p1, p2, goban, verbose=False)

        if winner == color1:
            win1 += 1
        elif winner == color2:
            win2 += 1

    # Turn verbosity on
    sys.stdout = sys.__stdout__
    return win1 / rounds


if __name__ == "__main__":
    'Train mcts using self play'
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()

    tf.get_logger().setLevel('INFO')

    saves_path_best_folder = os.path.join("saves", args.name, "best")
    os.makedirs(saves_path_best_folder, exist_ok=True)
    saves_path_best = os.path.join(saves_path_best_folder, "model")

    saves_path_current_folder = os.path.join(
        "saves", args.name, "current")
    os.makedirs(saves_path_current_folder, exist_ok=True)
    saves_path_current = os.path.join(saves_path_current_folder, "model")

    best_mcts = None
    for round_idx in range(1, 1+TRAIN_ROUNDS):
        print("Starting round: ", round_idx)
        starting_goban = Goban(GOBAN_SIZE)
        current_mcts = MCTS(GOBAN_SIZE)
        current_mcts.train(starting_goban, N_EPISODES,
                           N_EPOCHS, FIT_EVERY_EPISODE, BATCH_SIZE, learning_rate=LEARNING_RATE, verbose=True)

        if round_idx % EVALUATE_EVERY_ROUND == 0:
            if best_mcts is not None:
                print("End of round, evaluating mcts...")
                current_mcts.model.save_weights(saves_path_current)

                win_ratio = evaluate(
                    saves_path_current, saves_path_best, EVALUATION_ROUNDS)
                print("MCTS evaluated, win ratio: ", win_ratio)

            if best_mcts is None or win_ratio >= BEST_NET_WIN_RATIO:
                print("Current mcts is better than best, saving...")
                best_mcts = current_mcts
                # best_mcts.save(path=save_path)
                best_mcts.model.save_weights(saves_path_best)

    shutil.rmtree(saves_path_current_folder)

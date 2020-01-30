#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre
#############################################################
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

NUM_FILTERS = 64


class AlphaZeroModel(tf.keras.Model):

    def __init__(self, input_shape, actions_n):
        super(AlphaZeroModel, self).__init__()

        # Main layers (conv)
        self.input_conv = Conv2D(NUM_FILTERS, kernel_size=3,
                                 activation='relu', name='input_conv', input_shape=(1, 7, 7))
        self.max_pooling = MaxPool2D((5, 5), name='pool')
        self.flatten = Flatten()

        # Value head
        self.hidden_value = Dense(
            NUM_FILTERS, activation='relu', name='h_val_1')
        self.value = Dense(1, name="output_value", activation="softmax")

        # Prior head
        self.hidden_prior = Dense(
            NUM_FILTERS, activation='relu', name='h_prior_1')
        self.prior = Dense(1, activation='relu', name='output_prior')

    def call(self, x):
        x = self.input_conv(x)
        x = self.max_pooling(x)
        x = self.flatten(x)

        # Value
        value = self.hidden_value(x)
        value = self.value(value)

        # Prior
        prior = self.hidden_prior(x)
        prior = self.prior(prior)

        return prior, value

    def fit(self, batch, epochs=10, batch_size=32, **kwargs):
        'batch: list of tuple: (states, best_actions, values)'
        states, best_actions, values = list(zip(*batch))
        optimizer = Adam(kwargs)
        experiences_all = zip(states, best_actions, values)
        for e in range(epochs):
            states, true_best_actions, true_values = random.sample(
                experiences_all, batch_size)

            with tf.GradientTape() as tape:
                pred_probs, pred_values = self.model.predict(states)
                value_loss = tf.keras.losses.mean_squared_error(
                    true_values, pred_values)
                prob_loss = tf.keras.losses.categorical_crossentropy(
                    true_best_actions, pred_probs)
                total_loss = value_loss + prob_loss

            gradients = tape.gradient(
                total_loss, self.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, self.trainable_variables))

            print("Epoch", e, "ended. Probs loss:", prob_loss,
                  "Value loss:", value_loss, "Total loss:", total_loss)

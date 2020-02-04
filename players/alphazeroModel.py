#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Alphago reproduction IA course project, ENSEIRB-MATMECA
# COUTHOUIS Fabien - HACHE Louis - Heuillet Alexandre
#############################################################
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam

NUM_FILTERS = 64


class AlphaZeroModel(tf.keras.Model):

    def __init__(self, input_shape, actions_n, model_path=None):
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
        self.prob = Dense(actions_n, activation='relu', name='output_prior')

        if model_path is not None:
            self._restore(model_path)

    def _restore(self, path):
        print("Restoring weights from", path, "...")

        self.load_weights(path)

    def call(self, x):
        # Change data type to float32
        x = tf.cast(x, tf.float32)

        x = self.input_conv(x)
        x = self.max_pooling(x)
        x = self.flatten(x)

        # Value
        value = self.hidden_value(x)
        value = self.value(value)

        # Prior
        prob = self.hidden_prior(x)
        prob = self.prob(prob)

        return value, prob

    def predict_one(self, x):
        'Prediction on one state. Return (value,prob).'
        value, prob = self.predict(np.expand_dims(x, axis=0).astype('float32'))
        return value[0][0], prob[0]

    def transform_batch(self, batch):
        'Return (states, true_best_actions, true_values) '
        states, actions, values = list(zip(*batch))
        return np.array(states), np.array(actions), np.array(values)

    def fit(self, batch, epochs=10, batch_size=32, verbose=True, **kwargs):
        """
        Fit model on given batch

        Parameters: 
            batch: List of tuple: (states, game result (value), node values (probs))
            epochs: Number of epochs to train with (optional, default=10)
            batch_size: (optional, default=32)
        """
        optimizer = Adam(**kwargs)
        for e in range(epochs):
            batch_sample = random.sample(
                batch, batch_size)
            states, true_values, true_probs = self.transform_batch(
                batch_sample)

            with tf.GradientTape() as tape:
                pred_values, pred_probs = self(states)

                # Add a square to give more importance to the value loss
                value_loss = tf.math.square(tf.keras.losses.mean_squared_error(
                    true_values, pred_values))

                prob_loss = -tf.keras.losses.categorical_crossentropy(
                    true_probs, pred_probs)
                total_loss = value_loss + prob_loss

            gradients = tape.gradient(
                total_loss, self.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, self.trainable_variables))

        tf.print("Probs loss:", sum(prob_loss),
                 "Value loss:", sum(value_loss), "Total loss:", sum(total_loss))

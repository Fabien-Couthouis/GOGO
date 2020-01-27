import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation, Conv2D, MaxPool2D, Flatten


class AlphaZeroModel(tf.keras.Model):


def __init__(self, input_shape, hidden_size):
        super().__init__('mlp_policy')

        # Main network
        self.input = Conv2D(input_shape, (1, 1),
                            activation='relu', name='conv')
        self.max_pooling = MaxPool2D((5, 5), name='pool')
        self.flatten = Flatten()

        # Value
        self.hidden_value = Dense(
            hidden_size, activation='relu', name='h_val_1')
        self.value = Dense(1, name="output_value", activation="softmax")

        # Prior
        self.hidden_prior = Dense(
            hidden_size, activation='relu', name='h_prior_1')
        self.prior = Dense(1, activation='relu', name='output_prior')

    def call(self, x):
        x = self.input(x)
        x = self.max_pooling(x)
        x = self.flatten(x)

        # Value
        value = self.hidden_value(x)
        value = self.value(value)

        # Prior
        prior = self.hidden_prior(x)
        prior = self.prior(prior)

        return prior, value

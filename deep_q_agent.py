import random
import numpy as np
from collections import deque
from tensorflow import keras


class DQNAgent:
    def __init__(self, state_shape, action_size, epsilon, gamma, learning_rate):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.learning_rate = learning_rate

        self.local_model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.target_model.get_weights())

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(128, (4, 4), activation='relu',
                                input_shape=self.state_shape),
            keras.layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same'),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(
            lr=self.learning_rate))
        return model

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.local_model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.local_model.predict(next_state)[0]))
            target_f = self.local_model.predict(state)
            target_f[0][action] = target
            self.local_model.fit(state, target_f, epochs=1, verbose=0)
            self.decay_epsilon()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_model.set_weights(self.local_model.get_weights())

    def load(self, name):
        self.local_model.load_weights(name)

    def save(self, name):
        self.local_model.save_weights(name)

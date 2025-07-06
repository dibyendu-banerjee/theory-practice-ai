# ================================================================
# File: dqn_agent.py
# Description: This script demonstrates two reinforcement learning 
# approaches: (1) classic Q-learning using a Q-table, and 
# (2) Deep Q-Network (DQN) using Keras. It includes training loops 
# and model definitions for environments with discrete action spaces.
#
# Author: Dibyendu Banerjee
# ================================================================

import numpy as np

# -------------------------------
# Part 1: Classic Q-Learning
# -------------------------------

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
episodes = 1000

# Assume env and Q_table are predefined externally
# Example: env = gym.make("FrozenLake-v1")
# Q_table = np.zeros((env.observation_space.n, env.action_space.n))

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        # Exploration vs Exploitation
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q_table[state])  # Exploit

        # Take action
        next_state, reward, done, info = env.step(action)

        # Q-learning update rule
        Q_table[state, action] = Q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[state, action]
        )

        state = next_state

# -------------------------------
# Part 2: Deep Q-Network (DQN)
# -------------------------------

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Build a simple DQN model
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Assume env is defined and has .observation_space and .action_space
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = build_model(state_size, action_size)

# DQN training loop (simplified)
def train_dqn(episodes):
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False

        while not done:
            # Choose action
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])

            # Take action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Train on the observed transition
            target = reward
            if not done:
                target += discount_factor * np.amax(model.predict(next_state, verbose=0)[0])

            target_f = q_values
            target_f[0][action] = target

            model.fit(state, target_f, epochs=1, verbose=0)

            state = next_state

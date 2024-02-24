import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import os
import cv2
import joblib

print("Num GPUs Available: ", tf.config.list_physical_devices('GPU') , len(tf.config.list_physical_devices('GPU')))
with tf.device('/GPU:0'):
    env = gym.make("ALE/Pong-v5", difficulty=0)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    batch_size = 32
    n_episodes = 500
    num_frames = 1
    height = 84
    width = 84


    class DQNAgent:
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=5000)
            self.gamma = 0.99
            self.epsilon = 0.1
            self.epsilon_decay = 0.9995
            self.epsilon_min = 0.1
            self.learning_rate = 0.00025
            self.model = joblib.load('models/pong_model_387.pkl')
            self.target_model = joblib.load('models/pong_model_387.pkl')
            self.target_update_counter = 0

        def _build_model(self):
            model = Sequential()
            model.add(Conv2D(32, 8, strides=(4, 4), input_shape=(height, width, num_frames), activation='relu'))
            model.add(Conv2D(64, 4, strides=(2, 2), activation='relu'))
            model.add(Conv2D(64, 3, strides=(1, 1), activation='relu'))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(self.action_size, activation=None))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state):
            if np.random.rand() <= self.epsilon:
                self.act_values = []
                return random.randrange(self.action_size)
            act_values = self.model.predict(state[np.newaxis], verbose=0)
            self.act_values = act_values
            return np.argmax(act_values[0])

        def replay(self, batch_size):
            minibatch = random.sample(self.memory, batch_size)
            current_states = np.array([transition[0] for transition in minibatch])
            current_qs_list = self.model.predict(current_states, verbose=0)
            new_current_states = np.array([transition[3] for transition in minibatch])
            future_qs_list = self.target_model.predict(new_current_states, verbose=0)

            X = []
            y = []

            for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + self.gamma * max_future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                X.append(current_state)
                y.append(current_qs)

            self.model.fit(
                np.array(X),
                np.array(y),
                batch_size=batch_size,
                verbose=0,
                epochs=1)

            self.target_update_counter += 1
            
            if self.target_update_counter == 1000:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        def load(self, name):
            self.model.load_weights(name)

        def save(self, name):
            self.model.save_weights(name)

        @staticmethod
        def etl(previous_state, state):
            previous_state = agent.fix_state(previous_state)
            state = agent.fix_state(state)
            max_state = np.maximum(previous_state, state)
            gray_scale = cv2.cvtColor(max_state, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_scale, (84, 84)) / 255.0
            return resized[:, :, np.newaxis]


        @staticmethod
        def fix_state(state):
            if isinstance(state, tuple):
                state = state[0]
            return state

    agent = DQNAgent(state_size, action_size)
    done = False

    for episode in range(388, n_episodes):
        old_frame = agent.fix_state(env.reset())

        episode_reward = 0
        done = False
        processed_frame = agent.etl(old_frame, old_frame)
        action = agent.act(processed_frame)

        while not done:
            action = agent.act(processed_frame)
            new_frame, reward, done, truncate, info = env.step(action)
            new_processed_frame = agent.etl(old_frame, new_frame)
            agent.remember(processed_frame, action, reward, new_processed_frame, done)         
            episode_reward += reward
            old_frame = new_frame
            processed_frame = new_processed_frame

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        joblib.dump(agent.model, 'pong_model_' + str(episode) + '.pkl')

        print(f'episode: {episode}/{n_episodes}, score: {episode_reward}, e: {agent.epsilon}')

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import os
import cv2

env = gym.make("ALE/Pong-v5")

state_size = env.observation_space.shape
action_size = env.action_space.n
batch_size = 32
n_episodes = 10000
output_dir = 'model_output/pong'
k = 4

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), input_shape=((84, 84, 1))))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (4, 4)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(256))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis], verbose=0)))
            
            if isinstance(state, tuple):
                state = state[0]

            target_f = self.model.predict(state[np.newaxis], verbose=0)
            target_f[0][action] = target

            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    @staticmethod
    def etl(state):
        gray_state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_state, (84, 110))
        cropped_image = resized[26:110, :]
        return cropped_image

    @staticmethod
    def fix_state(state):
        if isinstance(state, tuple):
            state = state[0]
        return state

    @staticmethod
    def save_step(save_dict, c, sequence, action, cropped_image_next, done):
        save_dict[c] = {}
        save_dict[c]['sequence'] = sequence
        save_dict[c]['action'] = action
        save_dict[c]['next_state'] = cropped_image_next
        save_dict[c]['done'] = done
        return save_dict

    
agent = DQNAgent(state_size, action_size)
done = False

for e in range(n_episodes):
    state = agent.fix_state(env.reset())
    cropped_image = agent.etl(state)
    action = agent.act(cropped_image)

    total_reward = 0
    save_dict = {}
    c = 0
    stack_count = 0

    for i in range(5000):
        cropped_image = agent.etl(state)
        stacks = []

        while stack_count < 4:
            stacks.append(cropped_image)
            state, reward, done, truncate, info = env.step(action)
            cropped_image = agent.etl(agent.fix_state(state))
            stack_count += 1
        
        sequence = np.stack(stacks)
        stack_count = 0
        action = agent.act(sequence)

        next_state, reward, done, truncate, info = env.step(action)
        cropped_image_next = agent.etl(agent.fix_state(next_state))

        save_dict = agent.save_step(save_dict, c, sequence, action, cropped_image_next, done)
        
        total_reward += reward
        state = agent.fix_state(next_state)
        c += 1

        if reward != 0:
            for g in save_dict:
                saved_reward = reward * agent.gamma * (g / len(save_dict))
                agent.remember(
                    save_dict[g]['sequence'],
                    save_dict[g]['action'],
                    saved_reward,
                    save_dict[g]['next_state'],
                    save_dict[g]['done']
                    )
                c = 0
        if done:
            print(f'episode: {e}/{n_episodes}, score: {total_reward}, e: {agent.epsilon}')        
            break
                
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
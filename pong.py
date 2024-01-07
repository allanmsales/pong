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
batch_size = 64
n_episodes = 1000000
output_dir = 'model_output/pong'
k = 4
channels = 1
num_frames = 16
height = 84
width = 84

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), input_shape=(height, width, num_frames * channels)))
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
            self.log = 'RANDOM'
            return random.randrange(self.action_size)
        self.log = 'POLICY'
        act_values = self.model.predict(state[np.newaxis], verbose=0)
        self.act_values = act_values
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
        return cropped_image / 255

    @staticmethod
    def fix_state(state):
        if isinstance(state, tuple):
            state = state[0]
        return state

    @staticmethod
    def save_step(save_dict, c, sequence, action, reward, frame_stack, done):
        save_dict[c] = {}
        save_dict[c]['sequence'] = sequence
        save_dict[c]['action'] = action
        save_dict[c]['reward'] = reward
        save_dict[c]['next_state'] = frame_stack
        save_dict[c]['done'] = done
        return save_dict
    
agent = DQNAgent(state_size, action_size)
done = False

frame_stack = np.zeros((height, width, num_frames * channels), dtype=np.float32)
initial_state = agent.fix_state(env.reset())

for i in range(num_frames):
    processed_frame = agent.etl(initial_state)
    frame_stack[:, :, i * channels:(i + 1) * channels] = processed_frame[:, :, np.newaxis]
action = agent.act(initial_state)
total_reward = 0
save_dict = {}
c = 0


for episode in range(n_episodes):
    state = frame_stack
    if episode % k == 0:
        action = agent.act(state)

    next_state, reward, done, truncate, info = env.step(action)

    total_reward += reward

    frame_stack[:, :, :channels] = frame_stack[:, :, channels - 1:2 * channels - 1]
    frame_stack[:, :, -channels:] = agent.etl(next_state)[:, :, np.newaxis]

    save_dict = agent.save_step(save_dict, c, state, action, reward, frame_stack, done)

    c += 1
    if reward != 0:
        if agent.log == 'POLICY':
            print(f'REWARD: {reward}, ACTION: {action}, EPISODES: {c}, POLICY: {agent.act_values}')
        for g in save_dict:
            if c > 35:
                save_dict[g]['total_rewards'] = save_dict[g]['reward'] + 1
            else:
                save_dict[g]['total_rewards'] = save_dict[g]['reward']
            for item in range(len(save_dict)):
                if item > g:
                    save_dict[g]['total_rewards'] += save_dict[item]['reward'] * agent.gamma ** (item - g)
            agent.remember(
                save_dict[g]['sequence'],
                save_dict[g]['action'],
                save_dict[g]['total_rewards'],
                save_dict[g]['next_state'],
                save_dict[g]['done']
                )
        c = 0


    if done:
        print(f'episode: {episode}/{n_episodes}, score: {total_reward}, e: {agent.epsilon}')
        agent.replay(batch_size)
        initial_state = agent.fix_state(env.reset())
        frame_stack[:, :, :] = 0.0
        total_reward = 0

        for i in range(num_frames):
            processed_frame = agent.etl(initial_state)
            frame_stack[:, :, i * channels:(i + 1) * channels] = processed_frame[:, :, np.newaxis]
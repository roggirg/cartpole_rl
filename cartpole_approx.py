import gym
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import random

np.random.seed(0)

# Exploration value
epsilon = 1.0
exploration_decay = 0.99
min_epsilon = 0.1

# Discount Factor
gamma = 1.0

# find out when success
ave_steps_per_episode = np.zeros((100, 1))
idx_steps = -1

env = gym.make('CartPole-v0')
env.seed(0)


def q_model(observations, actions):
    model = Sequential()
    model.add(Dense(8, batch_input_shape=(None, observations), init='lecun_uniform', activation='relu'))
    model.add(Dense(8, init='lecun_uniform', activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model


def select_action(qs):
    # Epsilon-greedy action selection
    allow_exploration = np.random.rand()
    if allow_exploration > epsilon:
        return np.argmax(qs)
    else:
        return env.action_space.sample()


def final_agent_test():
    env = gym.make("CartPole-v0")
    for i_episode in range(10):
        state = env.reset()
        for t in range(500):
            env.render()
            action = np.argmax(q_vals.predict(np.reshape(state, (1, 4))))
            next_state, reward, done, info = env.step(action)
            state = next_state
            if reward == 0.0:
                break
        print("Episode {} finished after {} timesteps with final Reward = {}".format(i_episode + 1, t + 1, reward))

# Model Definition
q_vals = q_model(observations=4, actions=2)

# Experience arrays
experience_states = []
experience_values = []
experience_idx = -1
full_arrays = 0
max_size = 100
batch_size = 20


for i_episode in xrange(5000):
    state = env.reset()

    for t in xrange(199):

        # Value of current state's actions
        Q_state = q_vals.predict(np.reshape(state, (1, 4)))
        action = select_action(Q_state)

        # System response to my action
        next_state, reward, done, info = env.step(action)

        # value of taking the greedy action in the next state
        Q_next_state = q_vals.predict(np.reshape(next_state, (1, 4)))

        if done:
            reward = -100

        # Calculate TD Error
        if done:
            Q_state[0][action] = reward
        else:
            Q_state[0][action] = reward + gamma * np.max(Q_next_state[:])

        # Increment experience index
        experience_idx += 1
        # Put values in experience arrays
        if experience_idx < max_size and not full_arrays:
            experience_states.append(state)
            experience_values.append(Q_state[0])
        else:
            full_arrays = 1
            if experience_idx == max_size:
                experience_idx = 0
            experience_states[experience_idx] = state
            experience_values[experience_idx] = Q_state[0]
            batch = random.sample(range(max_size), batch_size)
            if experience_idx % 2 == 0:
                q_vals.train_on_batch(np.array(experience_states)[batch], np.array(experience_values)[batch])

        # Set new state as current state for next iter
        state = next_state

        if done or t == 198:
            if (i_episode + 1) % 10 == 0:
                print("Episode {} finished after {} timesteps | Epsilon = {}".format(i_episode+1, t+1, epsilon))
            break

    # Update exploration variable
    if epsilon > min_epsilon:
        epsilon *= exploration_decay
    else:
        epsilon = min_epsilon

    # Keeping the running average
    idx_steps += 1
    if idx_steps > 99:
        idx_steps = 0
    ave_steps_per_episode[idx_steps] = t+1
    if np.mean(ave_steps_per_episode) >= 195:
        final_agent_test()
        break

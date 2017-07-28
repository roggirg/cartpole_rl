import gym
import numpy as np

np.random.seed(0)

# Exploration value
epsilon = 1.0
exploration_decay = 0.99
min_epsilon = 0.01

# Discount Factor
gamma = 1.0

# Learning Rate
alpha = 0.2
alpha_decay = 0.9
min_alpha = 0.1

# find out when success
ave_steps_per_episode = np.zeros((100, 1))
idx_steps = -1

env = gym.make('CartPole-v0')
env.seed(0)
q_weights = np.random.rand(2, 4)


def q_vals(state_signal):
    return np.dot(q_weights, state_signal)


def select_action(qs):
    # Epsilon-greedy action selection
    allow_exploration = np.random.rand()
    if allow_exploration > epsilon:
        return np.argmax(qs)
    else:
        return env.action_space.sample()


def final_agent_test():
    env = gym.make("CartPole-v0")
    _epsilon = 0.0
    for i_episode in range(10):
        state = env.reset()
        for t in range(500):
            env.render()
            action = np.argmax(q_vals(state))
            next_state, reward, done, info = env.step(action)
            state = next_state
            if reward == 0.0:
                break
        print("Episode {} finished after {} timesteps with final Reward = {}".format(i_episode + 1, t + 1, reward))


for i_episode in xrange(5000):
    state = env.reset()

    # Value of current state's actions
    Q_state = q_vals(state)
    action = select_action(Q_state)

    for t in xrange(199):

        # Calculate value of next state for each action
        Q_state = q_vals(state)

        # System response to my action
        next_state, reward, done, info = env.step(action)

        # value of taking the greedy action in the next state
        Q_next_state = q_vals(next_state)
        next_action = select_action(Q_next_state)

        if done:
            reward = -200

        # Calculate TD Error
        if done:
            td_error = reward
        else:
            td_error = reward + (gamma * Q_next_state[next_action]) - Q_state[action]

        # Make update to weights
        q_weights[action] += alpha * td_error * state

        # Set new state as current state for next iter
        state = next_state
        action = next_action

        if done or t == 198:
            if (i_episode + 1) % 10 == 0:
                print("Episode {} finished after {} timesteps".format(i_episode+1, t+1))
            break

    # Update exploration variable
    if epsilon > min_epsilon:
        epsilon *= exploration_decay
    else:
        epsilon = min_epsilon

    # Decay learning rate up to a minimum
    if alpha > min_alpha:
        alpha *= alpha_decay
    else:
        alpha = min_alpha

    # Keeping the running average
    idx_steps += 1
    if idx_steps > 99:
        idx_steps = 0
    ave_steps_per_episode[idx_steps] = t + 1
    if np.mean(ave_steps_per_episode) >= 195:
        final_agent_test()
        break

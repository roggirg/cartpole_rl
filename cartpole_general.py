import gym
import numpy as np


class RL_Agent:
    def __init__(self, num_actions, func_approx=False, algorithm='Qlearn', et_lambda=0.5):
        self.env = gym.make("CartPole-v0")
        self.env.seed(0)
        self.num_episodes = 5000
        self.num_timesteps = 200
        self.num_actions = num_actions
        self.gamma = 1.0  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.minimum_exploration = 0.01
        self.exploration_decay = 0.99
        self.alpha = 0.5
        self.et_lambda = et_lambda  # Lambda for eligibility trace
        if not func_approx:
            self.create_bins()
        self.create_q_vals()
        self.algorithm = algorithm

    def create_bins(self):
        cart_pos_bins = np.array([-2.4, -1.2, -0.6, 0.0, 0.6, 1.2, 2.4])
        cart_vel_bins = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        pole_ang_bins = np.array([-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])
        pole_tipvel_bins = np.array([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5])
        self.bins = [cart_pos_bins, cart_vel_bins, pole_ang_bins, pole_tipvel_bins]

    def discretize_state(self, observation):
        state_signal = 0  # row position in Q-matrix or eligibility trace, equivalent state
        bin_state = np.zeros_like(observation)
        for i in range(len(observation)):
            bin_state[i] = np.digitize(observation[i], self.bins[i])
            state_signal += bin_state[i] * ((len(self.bins[i]) + 1) ** i)
        return int(state_signal)

    def create_q_vals(self):
        num_states = (len(self.bins[0]) + 1) * (len(self.bins[1]) + 1) * (len(self.bins[2]) + 1) * (len(self.bins[3]) + 1)
        self.Q_matrix = np.zeros((num_states, self.num_actions))

    def zero_eligibility_traces(self):
        self.eligibility_traces = np.zeros_like(self.Q_matrix)

    def eps_action_selection(self, state):
        allow_exploration = np.random.rand()
        if allow_exploration <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q_matrix[state, :])
        return action

    def next_action_value(self):
        if self.algorithm == 'Qlearn':
            self.Q_next_state = np.max(self.Q_matrix[self.next_state, :])
        elif self.algorithm == 'SARSA':
            self.next_action = self.eps_action_selection(self.next_state)
            self.Q_next_state = self.Q_matrix[self.next_state, self.next_action]

    def update_q_vals(self):
        td_error = self.reward + self.gamma * self.Q_next_state - self.Q_matrix[self.state, self.action]
        self.eligibility_traces[self.state, self.action] += 1
        self.Q_matrix += self.alpha * td_error * self.eligibility_traces
        self.eligibility_traces *= self.gamma * self.et_lambda

    def decay_epsilon(self):
        if self.epsilon > self.minimum_exploration:
            self.epsilon *= self.exploration_decay
        else:
            self.epsilon = self.minimum_exploration

    def train_agent(self):

        # Average Number of steps per episodes
        ave_steps_per_episode = np.zeros((100, 1))
        idx_steps = -1
        for i_episode in range(self.num_episodes):

            # Adding Eligibility traces
            self.zero_eligibility_traces()

            # Initialize state
            self.state = self.discretize_state(self.env.reset())

            # Choose Action based on epsilon greedy for SARSA OUTSIDE LOOP
            if self.algorithm == 'SARSA':
                self.action = self.eps_action_selection(self.state)

            for t in range(200):  # Maximum is 200 for game dynamics

                # Choose Action based on epsilon greedy for Q-learning INSIDE LOOP
                if self.algorithm == 'Qlearn':
                    self.action = self.eps_action_selection(self.state)

                # Take the action chosen above and observe state and reward
                next_state, self.reward, done, info = self.env.step(self.action)
                self.next_state = self.discretize_state(next_state)

                # Experimenting with very bad reward if agent fails
                if done and t < 199:
                    self.reward = -100

                # We need to check if SARSA or Q-learning to decide how to choose next action
                self.next_action_value()

                # Compute TD-error and eligibility traces for the current state-action pair
                # For all states and actions, update Q-values and eligibility trace
                self.update_q_vals()

                # Update the new state and new action to become the current state and action
                self.state = self.next_state
                if self.algorithm == 'SARSA':
                    self.action = self.next_action

                if done:
                    if (i_episode + 1) % 10 == 0:
                        print("Episode {} finished after {} timesteps".format(i_episode + 1, t + 1))
                    break

            # Update exploration variable
            self.decay_epsilon()

            # Keeping the running average to determine if the agent is doing good
            idx_steps += 1
            if idx_steps > 99:
                idx_steps = 0
            ave_steps_per_episode[idx_steps] = t
            if np.mean(ave_steps_per_episode) >= 195:  # As per the rules from Gym
                break

    def final_agent_test(self):
        for i_episode in range(10):
            self.state = self.discretize_state(self.env.reset())
            for t in range(500):

                # Render the environment
                self.env.render()

                # Choose the greedy action
                action = np.argmax(self.Q_matrix[self.state, :])

                # Take action and Observe next state and reward
                self.next_state, self.reward, done, info = self.env.step(action)

                self.state = self.discretize_state(self.next_state)

                # When the agent drops the pole or goes off screen, stop the episode
                if self.reward == 0.0:
                    break

            # Want to know at what timestep the agent messed up
            print("Episode {} finished after {} timesteps with final Reward = {}".format(i_episode + 1, t + 1, self.reward))

if __name__ == "__main__":
    np.random.seed(0)
    agent = RL_Agent(num_actions=2, et_lambda=0.3, algorithm='Qlearn')  # 2 for cartpole, left and right
    agent.train_agent()
    agent.final_agent_test()

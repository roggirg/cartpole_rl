# cartpole_rl

This project tackles the cartpole problem from OpenAi Gym using reinforcement learning techniques.

# Dependencies

All you need (for now):
- Python 2/3
- Numpy
- OpenAI Gym
- Keras

# Description

After going through the first six RL lectures by David Silver and first ten chapters of "Reinforcement Learning: 
An Introduction (Sutton and Barto)", I implemented the following three methods:
- SARSA and Q-learning algorithms with eligibility traces (adjustable lambda) for the tabular.
- SARSA and Q-learning algorithms with linear value function approximation (LVFA). (No Eligibility traces)
- Q-Learning algorithm with a neural network as the action-value function approximator (DQN). (Experience Replay)

# Details of Each Method
- Tabular Method: Since the cartpole problem is composed of continuous state-space features, for this method, I discretize it using the create_bins method. The bins have been arbitrarily chosen; the only thing I wanted to maintain was to keep them centered around zero.

- LVFA Method: I create a matrix of size (num_of_actions, size_of_state_signal) initialized randomly. When you take the dot product of this matrix with the state signal, we get the value for each action. Updating is done with stochastic gradient descent. As per the theory, the learning rate is decayed up to a minimum value. In the coming days, I may try to implement the TD(lambda) version of each algorithm.

- DQN Method: Here, I create a small 2-layer neural network, using keras, with an Adam optimizer and mean-squared error loss function. I maintain a 100 timesteps of the latest experience and train the agent every 2 timesteps on 20 randomly sampled timesteps. Of course, these values can (and should) be played with. Otherwise, this follows the Q-learning algorithm for action-selection.

# Playing with Hyperparameters (Empirical)
As per the rules by OpenAI, "Considered solved when the average reward is greater than or equal to 
195.0 over 100 consecutive trials." 

- Lambda for Eligibility Traces: The agent succeeds when et_lambda=[0.0, 0.6], whether SARSA or Q-learning.

- Discount Factor: The agent is capable of with very little discounting. That is, gamma=[0.99, 1.0], 
whether SARSA or Q-learning.

- Epsilon Exploration Factor: As far as I can tell, the best performance was achieved when the it starts at full 
exploration and decays with 0.99. I've also added a minimum_exploration of 0.01. This may, in some cases, cause the agent not 
to converge "fast", but the final performance is still achieved.

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

# Playing with Hyperparameters (Empirically)
In all implemented methods, I follow the rules set by OpenAI, i.e. "Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.". Therefore, when an agent achieves a mean score of 195.0 over 100 consecutive trials, I break from the training loop and test the agent using the final_agent_test() function. However, in the testing phase, I require that the agent survives 500 timesteps without failing 10 episodes.

After playing with all the hyperparameters, the following table shows the best hyperparameters that meet the 500-timestep goal.

| Algorithm | Method | Alpha | Lambda | Number of Episodes before success |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| SARSA  | Tabular | 0.45 Constant | 0.0  | 420 |
| Q-learning  | Tabular | 0.46 Constant | 0.0  | 450 |
| SARSA  | Tabular | 0.46 Constant | 0.37 | 360 |
| Q-learning  | Tabular | 0.5-->0.1 (0.9 decay) | 0.5 | 280 |
| SARSA  | LVFA | 0.2-->0.1 (0.9 decay) | N/A | 890 |
| Q-learning | LVFA | 0.5-->0.4 (0.99 decay) | N/A | 2010 |
| Q-learning | Neural Network | 0.001 (Adam) | N/A | 320 |

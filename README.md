# cartpole_rl

This project tackles the cartpole problem from OpenAi Gym using reinforcement learning techniques.

# Dependencies

All you need (for now):
- Python 2/3
- Numpy
- OpenAI Gym

# Description

After going through the first five RL lectures by David Silver and "Reinforcement Learning: 
An Introduction (Sutton and Barto)", I implemented SARSA and Q-learning algorithms with eligibility traces (adjustable lambda).

Since the cartpole problem is composed of continuous state-space features, for now, I discretize it using the 
create_bins method. As I go through the next chapters of the book and more David Silver lectures,
I will add a function approximation approach.

# Playing with Hyperparameters (Empirical)
As per the rules by OpenAI, "Considered solved when the average reward is greater than or equal to 
195.0 over 100 consecutive trials." 

- Lambda for Eligibility Traces: The agent succeeds when et_lambda=[0.0, 0.6], whether SARSA or Q-learning.

- Discount Factor: The agent is capable of with very little discounting. That is, gamma=[0.99, 1.0], 
whether SARSA or Q-learning.

- Epsilon Exploration Factor: As far as I can tell, the best performance was achieved when the it starts at full 
exploration and decays with 0.99. I've also added a minimum_exploration of 0.01. This may, in some cases, cause the agent not 
to converge "fast", but the final performance is still achieved.

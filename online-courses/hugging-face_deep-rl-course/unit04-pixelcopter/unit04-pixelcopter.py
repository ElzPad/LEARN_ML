import numpy as np

from collections import deque

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gym
import gym_pygame

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        # Define the three layers here
        self.fc1 = nn.Linear(s_size, h_size)

        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size*2, h_size)

        self.fc4 = nn.Linear(h_size, a_size)

    def forward(self, x):
        # Define the forward process here
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = F.softmax(self.fc4(x), dim=1)
        return x

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset() # DONE: reset the environment
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state) # DONE: get the action
            saved_log_probs.append(log_prob)
            next_state, reward, done, info = env.step(action) # DONE: take an env step
            state = next_state
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as the sum of the gamma-discounted return at time t (G_t) + the reward at time t

        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...


        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(rewards[t] + gamma * disc_return_t) # DONE: complete here

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()

        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            next_state, reward, done, info = env.step(action) # DONE: take an env step
            total_rewards_ep += reward

            if done:
                break
            state = next_state
        episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def main():
    # %% Explore environment
    # Create the env
    env_id = "Pixelcopter-PLE-v0"
    env = gym.make(env_id, disable_env_checker=True)
    env.reset()

    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    
    print("_______________OBSERVATION SPACE_______________")
    print("Observation Space Shape", env.observation_space.shape)
    print("Sample observation", env.observation_space.sample()) # Get a random observation

    print("\n_______________ACTION SPACE_______________")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample()) # Take a random action


    # %% Train the agent
    print("\n_______________Training_______________")
    pixelcopter_hyperparameters = {
        "h_size": 64,
        "n_training_episodes": 50000,
        "n_evaluation_episodes": 10,
        "max_t": 10000,
        "gamma": 0.99,
        "lr": 1e-4,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    np.bool8 = np.bool
    pixelcopter_policy = Policy(pixelcopter_hyperparameters["state_space"], pixelcopter_hyperparameters["action_space"], pixelcopter_hyperparameters["h_size"]).to(device)
    pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])

    reinforce(env,
            pixelcopter_policy,
            pixelcopter_optimizer,
            pixelcopter_hyperparameters["n_training_episodes"],
            pixelcopter_hyperparameters["max_t"],
            pixelcopter_hyperparameters["gamma"],
            print_every=1000)

    # %% Evaluate the agent
    # Create the evaluation env
    print("\n_______________Evaluating_______________")
    eval_env = gym.make(env_id)
    eval_mean, eval_std = evaluate_agent(eval_env,
                                        pixelcopter_hyperparameters["max_t"],
                                        pixelcopter_hyperparameters["n_evaluation_episodes"],
                                        pixelcopter_policy)
    print(f"Mean reward: {eval_mean:.2f} +/- {eval_std:.2f}")

if __name__ == "__main__":
    main()
import numpy as np

def simulate_mdp(P, R, policy, start_state, terminal_states, gamma, num_episodes, max_steps, seed=42):
    """
    Simulate episodes in an MDP and compute discounted returns.
    
    Args:
        P: Transition probabilities, shape (n_states, n_actions, n_states)
        R: Rewards, shape (n_states, n_actions, n_states)
        policy: Stochastic policy, shape (n_states, n_actions)
        start_state: Initial state for each episode
        terminal_states: List of terminal state indices
        gamma: Discount factor
        num_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (episode_returns, average_return)
    """
    np.random.seed(seed)

    n_states, n_actions = P.shape[0], P.shape[1]
    episode_returns = np.zeros(num_episodes)

    for episode in range(num_episodes):
        state = start_state
        ep_return = 0

        for t in range(max_steps):
            if state in terminal_states:
                break

            action = np.random.choice(n_actions, p=policy[state])
            next_state = np.random.choice(n_states, p=P[state][action])
            reward = R[state][action][next_state]

            state = next_state
            ep_return += gamma**t * reward


        episode_returns[episode] = np.round(ep_return, 4)

    return episode_returns.tolist(), np.round(np.mean(episode_returns), 4)

if __name__ == "__main__":
    P = np.array([[[0, 1, 0]], [[0, 0, 1]], [[0, 0, 1]]])
    R = np.array([[[0, 1, 0]], [[0, 0, 2]], [[0, 0, 0]]])
    policy = np.array([[1.0], [1.0], [1.0]])
    result = simulate_mdp(P, R, policy, 0, [2], 0.9, 3, 10, seed=42)
    print(result) # Expected output: [2.8, 2.8, 2.8] 2.8

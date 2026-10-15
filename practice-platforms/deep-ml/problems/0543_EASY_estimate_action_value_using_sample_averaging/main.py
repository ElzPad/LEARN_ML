def sample_average_action_values(k: int, actions: list, rewards: list) -> tuple:
    """
    Estimate action values using sample averaging.
    
    Args:
        k: Number of possible actions (labeled 0 to k-1)
        actions: List of actions taken at each time step
        rewards: List of rewards received at each time step
        
    Returns:
        Tuple of (Q, N) where Q is estimated values and N is selection counts
    """
    total_action_reward = [0] * k
    total_action_selection = [0] * k
    result_Q = []

    for t in range(len(actions)):
        total_action_reward[actions[t]] += rewards[t]
        total_action_selection[actions[t]] += 1

    for action in range(k):
        result_Q.append(0.0 if total_action_selection[action] == 0 else round(total_action_reward[action] / total_action_selection[action], 4))
    return result_Q, total_action_selection

if __name__ == "__main__":
    result = sample_average_action_values(2, [0, 1, 0, 1, 0], [1.0, 2.0, 3.0, 4.0, 5.0])
    print(result) # Expected output: ([3.0, 3.0], [3, 2])
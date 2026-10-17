import numpy as np

def softmax(preferences):
    # Compute softmax probabilities from preferences
    shifted_preferences = preferences - np.max(preferences)
    denominator = np.sum(np.exp(shifted_preferences))
    return np.exp(shifted_preferences) / denominator

def epsilon_greedy(
    true_values: list,
    reward_noise: list,
    epsilon: float,
) -> dict:
    time_length = len(reward_noise)
    optimal_arm = np.argmax(true_values)

    n_arms = len(reward_noise[0])
    q_values = np.zeros(n_arms)
    action_count = np.zeros(n_arms)

    optimal_selections = 0
    total_reward = 0
    
    for time in range(time_length):
        if np.random.rand() < epsilon:
            action = np.random.randint(n_arms)
        else:
            action = np.argmax(q_values)
        
        optimal_selections += 1 if action == optimal_arm else 0
        reward = true_values[action] + reward_noise[time][action]

        total_reward += reward
        action_count[action] += 1
        q_values[action] += (reward - q_values[action]) / action_count[action]

    result = {
        'avg_reward': np.round(total_reward / time_length, 4),
        'pct_optimal': np.round(optimal_selections / time_length * 100, 2) ,
    }
    return result

def ubc(
    true_values: list,
    reward_noise: list,
    ucb_c: float,  
) -> dict:
    time_length = len(reward_noise)
    optimal_arm = np.argmax(true_values)

    n_arms = len(reward_noise[0])
    q_values = np.zeros(n_arms)
    action_count = np.zeros(n_arms)

    optimal_selections = 0
    total_reward = 0
    
    for time in range(time_length):
        ucb_values = q_values + ucb_c * np.sqrt(np.log(time) / action_count)
        action = np.argmax(ucb_values)
        
        optimal_selections += 1 if action == optimal_arm else 0
        reward = true_values[action] + reward_noise[time][action]

        total_reward += reward
        action_count[action] += 1
        q_values[action] += (reward - q_values[action]) / action_count[action]

    result = {
        'avg_reward': np.round(total_reward / time_length, 4),
        'pct_optimal': np.round(optimal_selections / time_length * 100, 2) ,
    }
    return result

def gradient(
    true_values: list,
    reward_noise: list,
    alpha: float,  
) -> dict:
    time_length = len(reward_noise)
    optimal_arm = np.argmax(true_values)

    n_arms = len(reward_noise[0])
    preferences = np.zeros(n_arms)

    optimal_selections = 0
    avg_reward = 0
    
    for time in range(time_length):
        probabilities = softmax(preferences)
        action = np.random.choice(n_arms, p=probabilities)
        
        optimal_selections += 1 if action == optimal_arm else 0
        reward = true_values[action] + reward_noise[time][action]


        discrete_kronecker = np.zeros_like(preferences)
        discrete_kronecker[action] = 1
        updates = alpha * (reward - avg_reward) * (discrete_kronecker - softmax(preferences))
        preferences += updates

        avg_reward += (reward - avg_reward) / (time+1)

    result = {
        'avg_reward': np.round(avg_reward, 4),
        'pct_optimal': np.round(optimal_selections / time_length * 100, 2) ,
    }
    return result

def compare_bandit_algorithms(
    true_values: list,
    reward_noise: list,
    epsilon: float,
    ucb_c: float,
    gradient_alpha: float,
    seed: int
) -> dict:
    """
    Run epsilon-greedy, UCB, and gradient bandit on the same problem instance.

    Args:
        true_values: True action values for each arm, length k.
        reward_noise: Noise table of shape (steps, k).
        epsilon: Exploration rate for epsilon-greedy.
        ucb_c: Confidence parameter for UCB.
        gradient_alpha: Step-size for gradient bandit.
        seed: Random seed for action selection.

    Returns:
        Dictionary with keys 'epsilon_greedy', 'ucb', 'gradient',
        each containing 'avg_reward' and 'pct_optimal'.
    """
    np.random.seed(seed)

    result = {
        'epsilon_greedy': epsilon_greedy(true_values, reward_noise, epsilon),
        'ucb': ubc(true_values, reward_noise, ucb_c),
        'gradient': gradient(true_values, reward_noise, gradient_alpha),
    }

    return result

if __name__ == "__main__":
    np.random.seed(99)
    noise = np.random.randn(500, 3).tolist()
    result = compare_bandit_algorithms([1.0, 0.0, -1.0], noise, 0.1, 2.0, 0.1, 42)

    print(result['epsilon_greedy']['avg_reward'])   # Expected output: 0.934 
    print(result['epsilon_greedy']['pct_optimal'])  # Expected output: 93.2
    print(result['ucb']['avg_reward'])              # Expected output: 0.9793
    print(result['ucb']['pct_optimal'])             # Expected output: 95.4
    print(result['gradient']['avg_reward'])         # Expected output: 0.9208
    print(result['gradient']['pct_optimal'])        # Expected output: 91.8
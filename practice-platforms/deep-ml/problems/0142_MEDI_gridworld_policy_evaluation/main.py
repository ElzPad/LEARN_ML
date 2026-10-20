def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats
    """
    size = 5
    V = [[0.0 for _ in range(size)] for _ in range(size)]

    terminal_states = {(0, 0), (0, 4), (4, 0), (4, 4)}

    actions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }

    while True:
        delta = 0.0
        new_V = [row[:] for row in V]

        for r in range(size):
            for c in range(size):
                if (r, c) in terminal_states:
                    continue

                value = 0.0

                for action, prob in policy[(r, c)].items():
                    dr, dc = actions[action]
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < size and 0 <= nc < size):
                        nr, nc = r, c

                    reward = -1
                    value += prob * (reward + gamma * V[nr][nc])

                new_V[r][c] = value
                delta = max(delta, abs(value - V[r][c]))

        V = new_V

        if delta < threshold:
            break
    return V

if __name__ == "__main__":
    grid_size = 5
    gamma = 0.9
    threshold = 0.001
    policy = {(i, j): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25} for i in range(grid_size) for j in range(grid_size)}
    V = gridworld_policy_evaluation(policy, gamma, threshold)
    result = [round(V[2][2], 4), V[0][0], V[0][4], V[4][0], V[4][4]]
    print(result) # Expected output: [-7.0902, 0.0, 0.0, 0.0, 0.0]
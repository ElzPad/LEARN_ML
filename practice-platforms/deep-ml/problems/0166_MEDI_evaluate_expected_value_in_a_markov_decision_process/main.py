import numpy as np

def expected_action_value(state, action, P, R, V, gamma):
    """
    Computes the expected value of taking `action` in `state` for the given MDP.
    Args:
      state: int or str, the current state
      action: str, the chosen action
      P: dict of dicts, P[s][a][s'] = prob of next state s' if a in s
      R: dict of dicts, R[s][a][s'] = reward for (s, a, s')
      V: np.ndarray, the value function vector, indexed by state
      gamma: float, discount factor
    Returns:
      float: expected value
    """
    expected_value = 0
    for next_state, prob in P[state][action].items():
        reward = R[state][action][next_state]
        expected_value += prob * (reward + gamma * V[next_state])
    return expected_value

if __name__ == "__main__":
    states = [0, 1]
    actions = ['a', 'b']
    P = {
        0: {'a': {0: 0.5, 1: 0.5}, 'b': {0: 1.0}},
        1: {'a': {1: 1.0}, 'b': {0: 0.7, 1: 0.3}}
    }
    R = {
        0: {'a': {0: 5, 1: 10}, 'b': {0: 2}},
        1: {'a': {1: 0}, 'b': {0: -1, 1: 3}}
    }
    gamma = 0.9
    V = np.array([1.0, 2.0])
    result = round(expected_action_value(0, 'a', P, R, V, gamma), 4)
    print(result) # Expected output: 8.85
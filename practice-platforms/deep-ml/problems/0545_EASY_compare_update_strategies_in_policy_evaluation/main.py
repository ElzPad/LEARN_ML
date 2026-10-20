import numpy as np

def compare_policy_evaluation(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float, n_sweeps: int, method: str) -> np.ndarray:
    """
    Perform policy evaluation sweeps using the specified update method.
    
    Args:
        V: np.ndarray of shape (n_states,), initial value function
        P: np.ndarray of shape (n_states, n_states), transition probability matrix under the policy
        R: np.ndarray of shape (n_states,), expected immediate reward for each state
        gamma: float, discount factor
        n_sweeps: int, number of sweeps to perform
        method: str, either 'synchronous' or 'in_place'
    
    Returns:
        np.ndarray: updated value function after n_sweeps
    """
    n_states = V.shape[0]

    for i in range(n_sweeps):
        if method == "synchronous":
            updated_V = np.zeros_like(V)
            for s in range(n_states):
                updated_V[s] = np.dot(P[s, :], R[s] + gamma*V)
            V = updated_V

        else:
            for s in range(n_states):
                V[s] = np.dot(P[s, :], R[s] + gamma*V)
    return V

if __name__ == "__main__":
    V = np.array([0.0, 0.0])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    R = np.array([2.0, 0.0])
    result = [round(float(v), 4) for v in compare_policy_evaluation(V, P, R, 0.9, 1, 'synchronous')]
    print(result) # Expected output: [2.0, 0.0]
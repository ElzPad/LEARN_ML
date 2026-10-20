import numpy as np

def bellman_expectation_value(P, R, policy, gamma):
    """
    Compute the state-value function V^pi for a given policy using
    the Bellman expectation equation.
    
    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states)
        R: Rewards, shape (num_states, num_actions, num_states)
        policy: Stochastic policy, shape (num_states, num_actions)
        gamma: Discount factor
    
    Returns:
        State-value function as numpy array of shape (num_states,)
    """
    num_states = P.shape[0]

    # Compute policy-weighted transition matrix P^pi
    P_pi = np.einsum('sa,san->sn', policy, P)

    # Compute policy-weighted reward vector r^pi
    r_pi = np.einsum('sa,san,san->s', policy, P, R)

    #  Construct (I - gamma * P^pi)
    I = np.eye(num_states)
    A = I - gamma * P_pi

    # Solve (I - gamma * P^pi) V^pi = r^pi
    V = np.linalg.solve(A, r_pi)

    return V

if __name__ == "__main__":
    P = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
    R = np.array([[[1.0, 0.0]], [[0.0, 2.0]]])
    policy = np.array([[1.0], [1.0]])
    result = np.round(bellman_expectation_value(P, R, policy, 0.9), 4).tolist()
    print(result)
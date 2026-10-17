import numpy as np

class GradientBandit:
    def __init__(self, num_actions, alpha=0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = np.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0
    
    def softmax(self):
        # Compute softmax probabilities from preferences
        shifted_preferences = self.preferences - np.max(self.preferences)
        denominator = np.sum(np.exp(shifted_preferences))
        return np.exp(shifted_preferences) / denominator
    
    def select_action(self):
        # Sample an action according to the softmax distribution
        softmax = self.softmax()
        return np.random.choice(len(softmax), p=softmax)
    
    def update(self, action, reward):
        # Update action preferences using the gradient ascent update
        self.time += 1
        self.avg_reward += (reward - self.avg_reward) / self.time
        
        discrete_kronecker = np.zeros_like(self.preferences)
        discrete_kronecker[action] = 1
        updates = self.alpha * (reward - self.avg_reward) * (discrete_kronecker - self.softmax())
        self.preferences += updates


if __name__ == "__main__":
    np.random.seed(0)
    gb = GradientBandit(num_actions=3, alpha=0.1)
    a1 = gb.select_action()
    gb.update(a1, reward=1.0)
    a2 = gb.select_action()
    gb.update(a2, reward=0.0)
    print(a1, a2) # Expected output = 1 2
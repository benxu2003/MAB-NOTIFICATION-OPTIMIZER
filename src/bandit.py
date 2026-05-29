import numpy as np

class ThompsonSamplingBandit:
    """
    Thompson Sampling Multi-Armed Bandit using conjugate Beta-Binomial distributions.
    """
    def __init__(self, variant_ids):
        self.variant_ids = variant_ids
        self.alphas = {v_id: 1 for v_id in variant_ids}
        self.betas = {v_id: 1 for v_id in variant_ids}

    def select_variant(self):
        best_variant = None
        max_sample = -1
        for v_id in self.variant_ids:
            sample = np.random.beta(self.alphas[v_id], self.betas[v_id])
            if sample > max_sample:
                max_sample = sample
                best_variant = v_id
        return best_variant

    def update(self, variant_id, reward):
        if reward == 1:
            self.alphas[variant_id] += 1
        else:
            self.betas[variant_id] += 1

class EpsilonGreedyBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit algorithm.
    With probability epsilon, explore randomly; with probability 1-epsilon, exploit best option.
    """
    def __init__(self, variant_ids, epsilon=0.1):
        self.variant_ids = variant_ids
        self.epsilon = epsilon
        self.counts = {v_id: 0 for v_id in variant_ids}
        self.values = {v_id: 0.0 for v_id in variant_ids}

    def select_variant(self):
        if np.random.random() < self.epsilon:
            # Explore: choose random arm
            return np.random.choice(self.variant_ids)
        else:
            # Exploit: choose best arm based on current estimates
            best_value = max(self.values.values())
            best_variants = [v_id for v_id, value in self.values.items() if value == best_value]
            return np.random.choice(best_variants)

    def update(self, variant_id, reward):
        self.counts[variant_id] += 1
        n = self.counts[variant_id]
        value = self.values[variant_id]
        # Incremental update formula
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[variant_id] = new_value

class UCB1Bandit:
    """
    Upper Confidence Bound (UCB1) Multi-Armed Bandit algorithm.
    Balances exploration and exploitation using confidence bounds.
    """
    def __init__(self, variant_ids):
        self.variant_ids = variant_ids
        self.counts = {v_id: 0 for v_id in variant_ids}
        self.values = {v_id: 0.0 for v_id in variant_ids}
        self.total_counts = 0

    def select_variant(self):
        # First, play each arm once if not all arms have been played
        for v_id in self.variant_ids:
            if self.counts[v_id] == 0:
                return v_id
        
        # Calculate UCB values for all arms
        ucb_values = {}
        for v_id in self.variant_ids:
            bonus = np.sqrt((2 * np.log(self.total_counts)) / float(self.counts[v_id]))
            ucb_values[v_id] = self.values[v_id] + bonus
        
        # Return arm with highest UCB value
        return max(ucb_values, key=ucb_values.get)

    def update(self, variant_id, reward):
        self.counts[variant_id] += 1
        self.total_counts += 1
        n = self.counts[variant_id]
        value = self.values[variant_id]
        # Incremental update formula
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[variant_id] = new_value
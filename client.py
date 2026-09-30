"""Generalized Advantage Estimation (GAE).
100% Python Standard Library.
"""

class GAEAdvantageEstimator:
    """Computes Generalized Advantage Estimation (GAE) across episodic trajectories."""
    @staticmethod
    def compute_gae(rewards: list, values: list, gamma: float = 0.99, lam: float = 0.95) -> dict:
        T = len(rewards)
        advantages = [0.0] * T
        last_gae = 0.0

        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_val - values[t]
            advantages[t] = delta + gamma * lam * last_gae
            last_gae = advantages[t]

        returns = [round(advantages[t] + values[t], 4) for t in range(T)]
        return {
            "advantages": [round(a, 4) for a in advantages],
            "returns": returns,
            "mean_advantage": round(sum(advantages) / max(T, 1), 4)
        }

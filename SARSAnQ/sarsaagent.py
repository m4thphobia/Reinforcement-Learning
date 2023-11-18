import numpy as np


def bins(clip_min: float, clip_max, num: float) -> np.ndarray:
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def discretize_state(observation: np.ndarray, num_discretize: int) -> int:
    c_pos, c_v, p_angle, p_v = observation
    discretized = [
        np.digitize(c_pos, bins=bins(-2.4, 2.4, num_discretize)),
        np.digitize(c_v, bins=bins(-3.0, 3.0, num_discretize)),
        np.digitize(p_angle, bins=bins(-0.5, 0.5, num_discretize)),
        np.digitize(p_v, bins=bins(-2.0, 2.0, num_discretize)),
    ]
    return sum([x * (num_discretize**i) for i, x in enumerate(discretized)])


class SarsaAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        num_discretize: int,
        gamma: float = 0.99,
        alpha: float = 0.5,
        epsilon: float = 0.7,
        max_initial_q: float = 0.1,
    ) -> None:

        self.action_size = action_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.qtable = np.random.uniform(
            low=-max_initial_q,
            high=max_initial_q,
            size=(num_discretize**state_size, action_size),
        )

    def get_action(self, state: int, episode: int) -> int:
        action_values = [self.qtable[state, a] for a in range(self.action_size)]
        max_action = np.argmax(action_values)
        base_prob = self.epsilon/self.action_size/(episode+1)
        action_probs = [base_prob for _ in range(self.action_size)]  # {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
        action_probs[max_action] += 1 - self.epsilon/(episode+1)
        return np.random.choice(range(self.action_size), p=action_probs)

    def get_greedy_action(self, state: int) -> int:
        action = np.argmax(self.qtable[state])
        return action

    def update_qtable(
        self, state: int, action: int, reward: int, next_state: int, next_action: int
    ) -> None:

        target = reward + self.gamma*self.qtable[next_state, next_action]
        self.qtable[state, action] += (target - self.qtable[state, action])*self.alpha

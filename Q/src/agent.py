import numpy as np
from collections import defaultdict, deque

class QAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        num_discretize: int,
        gamma: float = 0.99,
        alpha: float = 0.5,
        epsilon: float = 0.7,
        max_initial_q: float = 1,
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
        base_prob = self.epsilon/self.action_size/(episode+1)
        if base_prob <= np.random.uniform(0, 1):
            action = np.argmax(self.qtable[state])
        else:
            action = np.random.choice([0, 1])
        return action

    def get_greedy_action(self, state: int) -> int:
        action = np.argmax(self.qtable[state])
        return action

    def  update_qtable(self, state: np.ndarray, action: int, reward:int, done: bool, next_state: np.ndarray) -> None:

        next_q = 0 if done else max(self.qtable[next_state])
        target = reward + self.gamma*next_q
        self.qtable[state, action] += (target - self.qtable[state, action])*self.alpha

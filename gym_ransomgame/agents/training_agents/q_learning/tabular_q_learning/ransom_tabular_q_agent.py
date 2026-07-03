from typing import Any

import numpy as np

from gym_idsgame.agents.training_agents.q_learning.q_agent import QAgent
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_ransomgame.envs import RansomGameEnv


class RansomTabularQAgent(QAgent):
    """
    RansomGame-specific Tabular Q-learning agent.

    This agent expects a RansomGameEnv and uses ransomgame_config instead of idsgame_config.
    """

    def __init__(self, env: RansomGameEnv, config: QAgentConfig):
        super().__init__(env, config)

        self.env: RansomGameEnv = env
        self.ransom_env: RansomGameEnv = env

        self.observation_to_state_id: dict[tuple, int] = {}

        self.Q_attacker = np.zeros((self.env.num_states_full, self.env.num_attack_actions))
        self.Q_defender = np.zeros((1, self.env.num_defense_actions))

        self.env.ransomgame_config.save_trajectories = False
        self.env.ransomgame_config.save_attack_stats = True

    def get_state_id(self, observation: Any) -> int:
        """
        Convert a RansomGame attacker observation into a stable integer state id.
        """
        state_key = (
            int(observation["time"]),
            tuple(int(x) for x in observation["stages"]),
        )

        if state_key not in self.observation_to_state_id:
            next_state_id = len(self.observation_to_state_id)

            if next_state_id >= self.Q_attacker.shape[0]:
                raise RuntimeError(
                    "RansomTabularQAgent discovered more states than Q_attacker was initialized for. "
                    "Increase env.num_states_full or switch Q_attacker to a dictionary-based table."
                )

            self.observation_to_state_id[state_key] = next_state_id

        return self.observation_to_state_id[state_key]

    def get_action(self, s: int, eval: bool = False, attacker: bool = True) -> int:
        """
        Sample an action using epsilon-greedy policy.
        """
        if attacker:
            actions = list(range(self.env.num_attack_actions))
            legal_actions = [
                action for action in actions
                if self.env.attacker_action_space.contains(action)
            ]
            q_table = self.Q_attacker
        else:
            actions = list(range(self.env.num_defense_actions))
            legal_actions = [
                action for action in actions
                if self.env.defender_action_space.contains(action)
            ]
            q_table = self.Q_defender

        if not legal_actions:
            raise AssertionError("No legal actions available")

        if (np.random.rand() < self.config.epsilon and not eval) or (
            eval and np.random.random() < self.config.eval_epsilon
        ):
            return int(np.random.choice(legal_actions))

        best_action = max(legal_actions, key=lambda action: q_table[s][action])
        return int(best_action)

    def q_learning_update(self, s: int, a: int, r: float, s_prime: int, attacker: bool = True) -> None:
        """
        Performs a Q-learning update.
        """
        if attacker:
            self.Q_attacker[s][a] = self.Q_attacker[s][a] + self.config.alpha * (
                r + self.config.gamma * np.max(self.Q_attacker[s_prime]) - self.Q_attacker[s][a]
            )
        else:
            self.Q_defender[s][a] = self.Q_defender[s][a] + self.config.alpha * (
                r + self.config.gamma * np.max(self.Q_defender[s_prime]) - self.Q_defender[s][a]
            )

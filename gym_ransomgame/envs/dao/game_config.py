"""
Game-specific configuration for the gym-idsgame environment
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from gym_idsgame.envs.dao.network_config import NetworkConfig

from typing import Optional
from gym_ransomgame.envs.dao.game_state import GameState


class GameConfig:
    """
    DTO with game configuration parameters
    """

    def __init__(
        self,
        initial_state: Optional[GameState] = None,
        ransomware: bool = False,
        manual_attacker: bool = False,
        manual_defender: bool = False,
        attacker: bool = True,
        defender: bool = False,
        num_attack_types: int = 10,
        max_value: int = 1,
        initial_state_path: Optional[str] = None,
        dense_rewards: bool = False,
    ):
        """
        Class constructor, initializes the DTO

        :param manual_attacker: whether the attacker is controlled manually or by an agent
        :param manual_attacker: whether the defender is controlled manually or by an agent
        :param num_attack_types: the number of attack types
        :param max_value: max value for a defense/attack attribute
        :param initial_state: the initial state
        :param initial_state_path: path to the initial state saved on disk
        :param dense_rewards: if true, give hacker dense rewards (reward for each intermediate server hacked)
        """
        self.ransomware = ransomware
        self.time = 0
        self.stages = np.zeros((1, 4), dtype=np.bool)
        self.exfiltration_level = 0
        self.encryption_level = 0
        self.benign_completed = 0
        self.local_detector_scores = np.zeros((1, 4))
        self.global_detector_score = 0
        # self.num_rows = 10
        # self.num_cols = 10
        self.manual_attacker = manual_attacker
        self.manual_defender = manual_defender
        self.attacker = attacker
        self.defender = defender
        self.num_attack_types = num_attack_types
        self.max_value = max_value
        self.num_attack_actions = 4  # 6
        self.num_defense_actions = 2
        self.num_states = 1
        # self.network_config = NetworkConfig(self.num_rows, self.num_cols, connected_layers=False)
        self.initial_state_path = initial_state_path
        # self.num_vulnerabilities_per_layer = None
        self.initial_state: GameState = initial_state  # type: ignore
        if self.initial_state is None and self.initial_state_path is not None:
            self.initial_state = GameState.load(self.initial_state_path)
        if self.initial_state is None:
            self.initial_state = GameState()
            self.initial_state.default_state(self.num_attack_types)
        self.dense_rewards = dense_rewards

    def set_load_initial_state(self, initial_state_path: str) -> None:
        """
        Sets the initial state by loading it from disk

        :param initial_state_path:
        :return: None
        """
        self.initial_state = GameState.load(initial_state_path)

    def set_initial_state(
        self,
        # time: int = 0,
        stages=None,
        exfiltration_level: int = 0,
        encryption_level: int = 0,
        percent_benign_completed: float = 0,
        local_detector_scores: Optional[np.ndarray] = None,
        global_detector_score: float = 0,
        **kwargs,
    ):
        """
        Utility function for setting the initial game state

        :param defense_val: defense value for defense types that are not vulnerable
        :param attack_val: attack value for attack types
        :return:
        """
        self.initial_state.set_state(
            # time=time,
            stages=np.zeros((1, 4), dtype=np.bool) if stages is None else stages,
            exfiltration_level=exfiltration_level,
            encryption_level=encryption_level,
            percent_benign_completed=percent_benign_completed,
            local_detector_scores=(
                np.zeros((1, 4))
                if local_detector_scores is None
                else local_detector_scores
            ),
            global_detector_score=global_detector_score,
        )

    def get_attacker_observation_space(self) -> gym.spaces.Dict:
        """
        Creates an OpenAI-Gym Space for the game observation

        :return: observation space
        """
        observation_space = gym.spaces.Dict(
            {
                # "time": Discrete(n=300, start=0, dtype=np.int32),
                "stages": gym.spaces.MultiBinary(n=4),
                "exfiltration_level": Discrete(n=GameState.N_PROGRESS_STEPS + 1),
                "encryption_level": Discrete(n=GameState.N_PROGRESS_STEPS + 1),
            }
        )
        return observation_space

    def get_defender_observation_space(self) -> gym.spaces.Dict:
        """
        Creates an OpenAI-Gym Space for the game observation

        :return: observation space
        """
        # TODO should record action history in defender observations
        # NOTE: kept in sync with GameState.get_defender_observation, which currently
        # emits only the encryption bar. Add exfiltration_level here (and there) to make
        # exfiltration progress visible to the defender.
        observation_space = gym.spaces.Dict(
            {
                "encryption_level": Discrete(n=GameState.N_PROGRESS_STEPS + 1),
                "global_detector_score": Discrete(n=GameState.N_PROGRESS_STEPS + 1),
            }
        )
        return observation_space

    def get_action_space(self, defender: bool = False) -> gym.spaces.Discrete:
        """
        Creates an OpenAi-Gym space for the actions in the environment

        :param defender: boolean flag if defender or not
        :return: action space
        """
        if defender:
            return gym.spaces.Discrete(self.num_defense_actions)
        else:
            return gym.spaces.Discrete(self.num_attack_actions)

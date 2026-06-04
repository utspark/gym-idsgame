"""
Game-specific configuration for the gym-idsgame environment
"""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from gym_ransomgame.envs.dao.game_state import GameState

class GameConfig():
    """
    DTO with game configuration parameters
    """
    def __init__(
            self,
            manual_attacker: bool = True,
            num_attack_types: int = 10,
            max_value: int = 9,
            initial_state: GameState = None,
            manual_defender: bool = False,
            initial_state_path :str = None,
            dense_rewards = False,
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
        self.percent_encrypted = None
        self.time = None
        self.manual_attacker = manual_attacker
        self.manual_defender = manual_defender
        self.num_attack_types = num_attack_types
        self.max_value = max_value
        self.num_attack_actions = 2
        self.num_defense_actions = 2
        self.num_states = 1
        self.initial_state_path = initial_state_path
        self.num_vulnerabilities_per_layer = None
        self.initial_state = initial_state
        if self.initial_state is None and self.initial_state_path is not None:
            self.initial_state = GameState.load(self.initial_state)
        if self.initial_state is None and self.initial_state_path is None:
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
            time=0,
            encrypted=0,
    ):
        """
        Utility function for setting the initial game state

        :param defense_val: defense value for defense types that are not vulnerable
        :param attack_val: attack value for attack types
        :return:
        """
        self.time = time
        self.percent_encrypted = encrypted
        self.initial_state.set_state(num_attack_types=self.num_attack_types)

    def get_attacker_observation_space(self) -> gym.spaces.Dict:
        """
        Creates an OpenAI-Gym Space for the game observation

        :return: observation space
        """
        observation_space = gym.spaces.Dict({
            "time": Discrete(n=200, start=0, dtype=np.int32),
            "stages": Discrete(n=4, start=0, dtype=np.int32)
        })
        return observation_space

    def get_defender_observation_space(self) -> gym.spaces.Dict:
        """
        Creates an OpenAI-Gym Space for the game observation

        :return: observation space
        """
        observation_space = gym.spaces.Dict({
            "time": Discrete(n=200, start=0, dtype=np.int32),
            "stages": gym.spaces.Box(low=0, high=10, shape=(1, 4), dtype=np.float32)
            # local detector and global detector threat scores
        })
        return observation_space

    def get_action_space(self, defender :bool = False) -> gym.spaces.Discrete:
        """
        Creates an OpenAi-Gym space for the actions in the environment

        :param defender: boolean flag if defender or not
        :return: action space
        """
        if defender:
            return gym.spaces.Discrete(self.num_defense_actions)
        else:
            return gym.spaces.Discrete(self.num_attack_actions)

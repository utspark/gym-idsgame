"""
Stateful data of the gym-ransomgame environment
"""
from typing import List, Optional, Tuple
import numpy as np
import pickle
from itertools import groupby

from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent

class GameState:
    """
    DTO representing the state of the game
    """

    def __init__(
        self,
        attack_values: Optional[np.ndarray] = None,
        defense_values: Optional[np.ndarray] = None,
        defense_det: Optional[np.ndarray] = None,
        attacker_pos: Tuple[int, int] = (0, 0),
        game_step: int = 0,
        attacker_cumulative_reward: int = 0,
        defender_cumulative_reward: int = 0,
        num_games: int = 0,
        attack_events: Optional[List[int]] = None,
        defense_events: Optional[List[AttackDefenseEvent]] = None,
        done: bool = False,
        detected: bool = False,
        attack_type: int = 0,
        num_hacks: int = 0,
        hacked: bool = False,
        np_random: Optional[np.random.Generator] = None,
        # min_random_a_val :int = 0,
        # min_random_d_val :int = 0,
        # min_random_det_val :int = 0,
        # max_value : int = 9,
        # reconnaissance_state : np.ndarray = None,
        # max_random_v_val = 2,
    ):
        """
        Constructor, initializes the DTO

        :param attack_values: the attack values for resource nodes in the network
        :param defense_values: the defense values for resource nodes in the network
        :param game_step: the number of steps of the current game
        :param attacker_cumulative_reward: the cumulative reward over all games of the attacker
        :param defender_cumulative_reward: the cumulative reward over all games of the defender
        :param num_games: the number of games played
        :param attack_events: attack events that are in queue to be simulated
        :param defense_events: defense events that are in queue to be simulated
        :param done: True if the game is over and otherwise False
        :param detected: True if the attacker is in a detected state, otherwise False
        :param attack_type: the type of the last attack
        :param num_hacks: number of wins for the attacker
        :param hacked: True if the attacker hacked the data node otherwise False
        """
        self.time = 0
        self.stages = np.zeros((1, 4))
        self.stage_time_spent = np.zeros((1, 4), dtype=int)
        self.percent_encrypted: float = 0
        self.percent_benign_completed: float = 0
        self.local_detector_scores: np.ndarray = np.zeros((1, 4))
        self.global_detector_score: float = 0

        if attack_values is None:
            attack_values = np.zeros((1, 1))
        self.attack_values: np.ndarray = attack_values

        if defense_values is None:
            defense_values = np.zeros((1, 1))
        self.defense_values: np.ndarray = defense_values

        self.defense_det: Optional[np.ndarray] = defense_det
        self.attacker_pos: Tuple[int, int] = attacker_pos
        # self.reconnaissance_state = reconnaissance_state
        self.game_step: int = game_step
        self.attacker_cumulative_reward: int = attacker_cumulative_reward
        self.defender_cumulative_reward: int = defender_cumulative_reward
        self.num_games: int = num_games

        if attack_events is None:
            attack_events = []
        self.attack_events: List[int] = attack_events

        if defense_events is None:
            defense_events = []
        self.defense_events: List[AttackDefenseEvent] = defense_events

        # self.min_random_a_val = min_random_a_val
        self.done = done
        self.detected = detected
        self.attack_defense_type = attack_type
        self.num_hacks = num_hacks
        self.hacked = hacked
        if np_random is None:
            np_random = np.random.default_rng()
        self.np_random: np.random.Generator = np_random
        self.action_descriptors = ["RE", "F1", "F2", "EX"]
        # self.reconnaissance_actions = []
        # self.max_random_v_val = max_random_v_val

    def default_state(
        self,
        num_attack_types: int,
        num_rows: int = 10,
        num_cols: int = 10,
        randomize_state : bool = False,
        randomize_visibility : bool = False,
        visibility_p : float = 0.5,
    ) -> None:
        """
        Creates a default state

        :param num_attack_types: the number of attack types
        :param randomize_state: boolean flag whether to create the state randomly
        :param randomize_state: boolean flag whether to create the state randomly
        :param randomize_visibility: boolean flag whether to randomize visibility for partially observed envs
        :return: None
        """
        self.set_state(
            0,
            np.zeros((1, 4)),
            0,
            0,
            np.zeros((1, 4)),
            0
        )
        self.defense_det = np.zeros((num_rows * num_cols, num_attack_types))
        self.defense_values = np.zeros((num_rows * num_cols, num_attack_types))
        self.attack_values = np.zeros((num_rows * num_cols, num_attack_types))
        self.attacker_pos = (0, 0)
        # self.attacker_pos = attacker_pos
        self.game_step = 0
        self.attacker_cumulative_reward = 0
        self.defender_cumulative_reward = 0
        self.num_games = 0
        self.attack_events = []
        self.defense_events = []
        self.done = False
        self.detected = False
        self.attack_defense_type = 0
        self.num_hacks = 0
        self.hacked = False


    def set_state(
            self,
            time: int,
            stages: Optional[np.ndarray] = None,
            percent_encrypted: float = 0,
            percent_benign_completed: float = 0,
            local_detector_scores: Optional[np.ndarray] = None,
            global_detector_score: float = 0,
    ):
        """
        Sets the state

        :param num_attack_types:  number of attack types
        :param det_val: detection value per node
        :param randomize_state: boolean flag whether to create the state randomly
        :param randomize_visibility: boolean flag whether to randomize visibility for partially observed envs
        :return: None
        """
        self.time: int = time
        if stages is None:
            stages = np.zeros((1, 4))
        self.stages: np.ndarray = stages
        self.percent_encrypted: float = percent_encrypted
        self.percent_benign_completed: float = percent_benign_completed
        if local_detector_scores is None:
            local_detector_scores = np.zeros((1, 4))
        self.local_detector_scores: np.ndarray = local_detector_scores
        self.global_detector_score: float = global_detector_score


    def new_game(
            self,
            init_state: "GameState",
            a_reward : int = 0,
            d_reward : int = 0,
            update_stats = True,
            randomize_state : bool = False,
            # network_config : NetworkConfig = None,
            num_attack_types : int = 0,
            # defense_val : int = None,
            # attack_val : int = None,
            # det_val : int = None,
            # vulnerability_val : int = None,
            # num_vulnerabilities_per_layer : int = None,
            # num_vulnerabilities_per_node : int = None,
            # randomize_visibility : bool = False,
            # visibility_p : float = 0.5,
            np_random: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Updates the current state for a new game

        :param init_state: the initial state of the first game
        :param a_reward: the reward delta to increment or decrement the attacker cumulative reward with
        :param d_reward: the reward delta to increment or decrement the defender cumulative reward with
        :param randomize_state: boolean flag whether to create the state randomly
        :param randomize_state: boolean flag whether to create the state randomly
        :return: None
        """
        if update_stats:
            self.num_games += 1
            if self.hacked:
                self.attacker_cumulative_reward += a_reward
                self.defender_cumulative_reward += d_reward
                self.num_hacks += 1
            if self.detected:
                self.attacker_cumulative_reward += a_reward
                self.defender_cumulative_reward += d_reward
        self.done = False
        self.time = 0
        self.attack_defense_type = 0
        self.game_step = 0
        self.attack_events = []
        self.defense_events = []
        if np_random is not None:
            self.np_random = np_random
        np_random = self.np_random
        if not randomize_state:
            self.attack_values = np.copy(init_state.attack_values)
            self.defense_values = np.copy(init_state.defense_values)
        else:
            self.set_state(
                num_attack_types,
            )
        self.detected = False
        self.hacked = False

    def copy(self) -> "GameState":
        """
        Creates a copy of the state

        :return: a copy of the current state
        """
        new_state = GameState()
        new_state.attack_values = np.copy(self.attack_values)
        new_state.defense_values = np.copy(self.defense_values)
        new_state.defense_det = np.copy(self.defense_det)
        # new_state.reconnaissance_state = np.copy(self.reconnaissance_state)
        new_state.time = self.time
        new_state.game_step = self.game_step
        new_state.attacker_cumulative_reward = self.attacker_cumulative_reward
        new_state.defender_cumulative_reward = self.defender_cumulative_reward
        new_state.num_games = self.num_games
        new_state.attack_events = self.attack_events
        new_state.defense_events = self.defense_events
        new_state.done = self.done
        new_state.detected = self.detected
        new_state.attack_defense_type = self.attack_defense_type
        new_state.num_hacks = self.num_hacks
        new_state.hacked = self.hacked
        new_state.np_random = self.np_random
        # new_state.reconnaissance_actions = self.reconnaissance_actions
        return new_state

    def attack(self, attack_type: int) -> None:
        """
        Implement this:
        Select some sort of attack action and time duration

        :param attack_type: the type of attack action to execute
        :return: None
        """
        self.add_attack_event(attack_type)
        return

    def defend(self, defense_type: int) -> bool:
        """
        Implement this:
        Raise the alarm or do nothing
        Or some sort of remediation action
        Or consider certain workloads as illegal

        :param defense_type: descriptor of defense class or approach?
        """
        return True

    @staticmethod
    def _calculate_exponential_probability(p_base: float, p_progress: float, n: int) -> float:
        """
        Calculates the probability using an exponential saturation model.
        p(n) = 1 - (1 - p_base) * (1 - p_progress)^(n-1)

        :param p_base: base probability for the first attempt (n=1)
        :param p_progress: progress rate per attempt
        :param n: number of consecutive attempts (n >= 1)
        :return: calculated probability
        """
        if n < 1:
            return 0.0
        return 1 - (1 - p_base) * (1 - p_progress) ** (n - 1)

    def simulate_attack(self, attack_type: int, np_random: Optional[np.random.Generator] = None) -> bool:
        """
        Simulates the outcome of an attack.

        Instead of a simple linear increase, this uses an exponential progress model
        to represent how repeated attempts increase the probability of success,
        modeling a 'work-to-completion' process.

        :param attack_type: the type of the attack
        :param np_random: random number generator
        :return: True if the attack was successful otherwise False
        """
        np_random = np_random or self.np_random
        assert np_random is not None

        # 1. Count consecutive occurrences of the same attack_type at the end of the history
        consecutive_attempts = 0
        for k, g in groupby(reversed(self.attack_events)):
            if k == attack_type:
                consecutive_attempts = len(list(g))
            break

        if consecutive_attempts == 0:
            return False

        # 2. Hardcoded parameters for each attack type: (p_base, p_progress)
        # 0: RE, 1: F1, 2: F2, 3: EX
        attack_configs = {
            0: (0.1, 0.1),
            1: (0.1, 0.1),
            2: (0.1, 0.1),
            3: (0.1, 0.1)
        }
        p_base, p_progress = attack_configs.get(attack_type, (0.1, 0.1))

        # 3. Calculate base probability using exponential saturation
        p = self._calculate_exponential_probability(p_base, p_progress, consecutive_attempts)

        # 4. Modifier: 3rd attack (index 2, "F2") gets a bonus if 2nd stage (index 1, "F1") is set to 1
        if attack_type == 2 and self.stages[0, 1] == 1:
            p += 0.4

        p = np.clip(p, 0, 1)
        return np_random.binomial(1, p) == 1

    def simulate_detection(self, np_random: Optional[np.random.Generator] = None) -> bool:
        """
        Implement this:

        :param node_id: the id of the node to simulate deteciton of
        :param reconnaissance: boolean flag, if true simulate detection of reconnaissance activity
        :param np_random: random number generator
        :return: True if the node was detected, otherwise False
        """
        np_random = np_random or self.np_random
        assert np_random is not None

        p = np.sum(self.stage_time_spent) / 100

        return np_random.binomial(1, p) == 1

    def defense_score(self, game_config):
        if not game_config.ransomware:
            return -1
        else:
            return 1 - self.percent_encrypted

    def attack_score(self, game_config, attack_action):
        if not game_config.ransomware:
            raise ValueError("Ransomware is not enabled")
        else:
            attack_reward = -0.1
            if attack_action == 0:
                attack_reward += 1
            elif attack_action == 1:
                attack_reward += 1
            elif attack_action == 2:
                attack_reward += 1
            elif attack_action == 3:
                attack_reward += 10
            else:
                raise ValueError("Invalid attack action")

            # + game_state.percent_encrypted + game_state.stages[0, 2] + game_state.stages[0, 3])

            return attack_reward

    def get_attacker_observation(self) -> dict:
        """
        Converts the state of the dynamical system into an observation for the attacker. As the environment
        is a partially observed markov decision process, the attacker observation is only a subset of the game state

        :param local_view: boolean flag indicating whether observations are provided in a local view or not
        :return: An observation of the environment
        """
        attacker_observation = {
            "time": int(self.time),
            "stages": self.stages.flatten().astype(np.int8)
        }
        return attacker_observation

    def get_attacker_node_from_observation(self, observation: np.ndarray, reconnaissance : bool = False) -> int:
        """
        Extracts which node the attacker is currently at from the observation representation

        :param observation: the observation representation emitted from the environment
        :param reconnaissance: boolean flag indicating whether the observation is from an env with reconnaissance state
        :return: the id of the node that the attacker is in
        """

        for node_id in range(len(observation)):
            if not reconnaissance:
                if observation[node_id][-1] == 1:
                    return node_id
            else:
                if observation[node_id][self.attack_values.shape[1]] == 1:
                    return node_id
        raise AssertionError("Could not find the node that the attacker is in")

    def add_attack_event(self, attack_type: int) -> None:
        """
        Adds an attack event to the state

        :param attack_type: the type of the attack
        :return: None
        """
        self.attack_events.append(attack_type)

    def add_defense_event(self, target_pos: Tuple[int, int], defense_type: int) -> None:
        """
        Adds a defense event to the state

        :param target_pos: the position in the grid of the target node
        :param defense_type: the type of the defense
        :return: None
        """
        defense_event = AttackDefenseEvent(target_pos, defense_type) # type: ignore
        self.defense_events.append(defense_event)

    def get_defender_observation(self) -> dict:
        """
        Converts the state of the dynamical system into an observation for the defender. As the environment
        is a partially observed markov decision process, the defender observation is only a subset of the game state

        :return: An observation of the environment
        """
        defender_observation = {
            "time": int(self.time),
            "local_detector_scores": self.local_detector_scores.astype(np.float32),
            "global_detector_score": np.array([self.global_detector_score], dtype=np.float32)
        }

        return defender_observation

    def restart(self) -> None:
        """
        Resets the game state, clears up all the history
        :return: None
        """
        self.num_games = 0
        self.num_hacks = 0
        self.defender_cumulative_reward = 0
        self.attacker_cumulative_reward = 0

    @staticmethod
    def load(path):
        filehandler = open(path, 'rb')
        return pickle.load(filehandler)

    @staticmethod
    def save(path, state):
        filehandler = open(path + "/initial_state.pkl", 'wb')
        pickle.dump(state, filehandler)

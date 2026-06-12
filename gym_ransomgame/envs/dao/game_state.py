"""
Stateful data of the gym-ransomgame environment
"""
from typing import Union, List
import numpy as np
import pickle

from uno import Bool

from gym_idsgame.envs.dao.node_type import NodeType
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent
from gym_idsgame.envs.dao.network_config import NetworkConfig

class GameState:
    """
    DTO representing the state of the game
    """

    def __init__(
        self,
        attack_values: np.ndarray = np.zeros((1, 1)),
        defense_values: np.ndarray = np.zeros((1, 1)),
        defense_det: np.ndarray = None,
        attacker_pos: Union[int, int] = (0, 0),
        game_step: int = 0,
        attacker_cumulative_reward: int = 0,
        defender_cumulative_reward: int = 0,
        num_games: int = 0,
        attack_events: List[AttackDefenseEvent] = None,
        defense_events: List[AttackDefenseEvent] = None,
        done: bool = False,
        detected: bool = False,
        attack_type: int = 0,
        num_hacks: int = 0,
        hacked: bool = False,
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
        self.percent_encrypted: float = 0
        self.percent_benign_completed: float = 0
        self.local_detector_scores: np.ndarray = np.zeros((1, 4))
        self.global_detector_score: float = 0

        self.attack_values = attack_values
        self.defense_values = defense_values
        self.defense_det = defense_det
        self.attacker_pos = attacker_pos
        # self.reconnaissance_state = reconnaissance_state
        self.game_step = game_step
        self.attacker_cumulative_reward = attacker_cumulative_reward
        self.defender_cumulative_reward = defender_cumulative_reward
        self.num_games = num_games
        self.attack_events = attack_events
        self.defense_events = defense_events
        # self.min_random_a_val = min_random_a_val
        # self.min_random_d_val = min_random_d_val
        # self.min_random_det_val = min_random_det_val
        # self.max_value = max_value
        if self.attack_events is None:
            self.attack_events = []
        if self.defense_events is None:
            self.defense_events = []
        self.done = done
        self.detected = detected
        self.attack_defense_type = attack_type
        self.num_hacks = num_hacks
        self.hacked = hacked
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
            stages: np.ndarray = np.zeros((1, 4)),
            percent_encrypted: float = 0,
            percent_benign_completed: float = 0,
            local_detector_scores: np.ndarray = np.zeros((1, 4)),
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
        self.stages = stages
        self.percent_encrypted = percent_encrypted
        self.percent_benign_completed = percent_benign_completed
        self.local_detector_scores = local_detector_scores
        self.global_detector_score = global_detector_score


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
            np_random: np.random.Generator = None,
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
        self.attack_defense_type = 0
        self.game_step = 0
        self.attack_events = []
        self.defense_events = []
        if np_random is None:
            np_random = np.random
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
        # new_state.reconnaissance_actions = self.reconnaissance_actions
        return new_state

    def attack(self, attack_type: int) -> None:
        """
        Implement this:
        Select some sort of attack action and time duration

        :param attack_type: the type of attack action to execute
        :return: None
        """
        pass

    def defend(self, defense_type: int) -> bool:
        """
        Implement this:
        Raise the alarm or do nothing
        Or some sort of remediation action
        Or consider certain workloads as illegal

        :param defense_type: descriptor of defense class or approach?
        """
        return True

    def simulate_attack(self, attack_type: int) -> bool:
        """
        Implement this:

        :param attacked_node_id: the id of the node that is attacked
        :param attack_type: the type of the attack
        :param network_config: NetworkConfig
        :return: True if the attack was successful otherwise False
        """
        return True

    def simulate_detection(self, node_id: int, np_random: np.random.Generator = None) -> bool:
        """
        Implement this:

        :param node_id: the id of the node to simulate deteciton of
        :param reconnaissance: boolean flag, if true simulate detection of reconnaissance activity
        :param np_random: random number generator
        :return: True if the node was detected, otherwise False
        """
        return True

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

    def add_attack_event(self, target_pos: Union[int, int], attack_type: int, attacker_pos: Union[int, int], reconnaissance: bool = False) -> None:
        """
        Adds an attack event to the state

        :param target_pos: position in the grid of the target node
        :param attack_type: the type of the attack
        :param attacker_pos: position of the attacker
        :param reconnaissance: reconnaissance flag
        :return: None
        """
        attack_event = AttackDefenseEvent(target_pos, attack_type, attacker_pos=attacker_pos, reconnaissance=reconnaissance)
        self.attack_events.append(attack_event)

    def add_defense_event(self, target_pos: Union[int, int], defense_type: int) -> None:
        """
        Adds a defense event to the state

        :param target_pos: the position in the grid of the target node
        :param defense_type: the type of the defense
        :return: None
        """
        defense_event = AttackDefenseEvent(target_pos, defense_type)
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

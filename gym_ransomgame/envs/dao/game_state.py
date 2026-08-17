"""
Stateful data of the gym-ransomgame environment
"""

from typing import List, Optional, Tuple, Dict
from types import MappingProxyType

import numpy as np
import pickle
from itertools import groupby

from gym_idsgame.envs.dao.attack_defense_event import AttackDefenseEvent


class GameState:
    """
    DTO representing the state of the game
    """

    # Stage Constants
    RECONNAISSANCE = 0
    COMPRESSION = 1
    EXFILTRATION = 2
    ENCRYPTION = 3
    IDLE = 4
    TERMINATE = 5

    # Progress Constants
    # The exfiltration/encryption progress bars are ordinal: a level in
    # [0, N_PROGRESS_STEPS] that never decreases within an episode. Stored as a scalar
    # rather than as a thermometer-coded bit array, since only N_PROGRESS_STEPS+1 of the
    # 2**N_PROGRESS_STEPS bit patterns are reachable.
    N_PROGRESS_STEPS = 4

    # Trip points for the exponential-saturation model, keyed on the attack probability
    PROGRESS_TRIPS = np.linspace(0.01, 0.501, N_PROGRESS_STEPS)

    # Trip points for the consecutive-attempt model, keyed on the completion ratio in
    # [0, 1]. Excludes 0 so that zero progress leaves the bar empty.
    CONSECUTIVE_PROGRESS_TRIPS = np.linspace(0, 1, N_PROGRESS_STEPS + 1)[1:]

    # Trip points for the global detector score bar, keyed on the raw detector score in
    # its theoretical range [0, 1]. Excludes 0 so that a zero score leaves the bar empty.
    DETECTOR_SCORE_TRIPS = np.linspace(0.01, 0.701, N_PROGRESS_STEPS)[1:]

    # Reward Constants
    DEFAULT_ATTACK_REWARD = -0.1
    DEFAULT_DEFENSE_REWARD = 0.1
    PROGRESS_REWARD = 0.2
    STAGE_REWARD = 0.5
    EXFILTRATION_REWARD = 1.5
    ENCRYPTION_REWARD = 2.5

    ATTACK_CONFIGS = MappingProxyType(
        # {
        #     RECONNAISSANCE: (0.1, 0.1),
        #     COMPRESSION: (0.1, 0.1),
        #     EXFILTRATION: (0.1, 0.1),
        #     ENCRYPTION: (0.1, 0.1),
        # }
        {
            RECONNAISSANCE: (0.2, 0.2),
            COMPRESSION: (0.2, 0.2),
            EXFILTRATION: (0.2, 0.2),
            ENCRYPTION: (0.2, 0.2),
        }
    )

    CONSECUTIVE_ATTEMPT_REQUIREMENTS = MappingProxyType(
        {
            RECONNAISSANCE: 2,
            COMPRESSION: 4,
            EXFILTRATION: 5,
            ENCRYPTION: 8,
        }
    )

    def __init__(
        self,
        attack_values: Optional[np.ndarray] = None,
        defense_values: Optional[np.ndarray] = None,
        defense_det: Optional[np.ndarray] = None,
        attacker_pos: Tuple[int, int] = (0, 0),
        game_step: int = 0,
        attacker_cumulative_reward: float = 0.0,
        defender_cumulative_reward: float = 0.0,
        num_games: int = 0,
        attack_events: Optional[List[int]] = None,
        defense_events: Optional[List[AttackDefenseEvent]] = None,
        done: bool = False,
        detected: bool = False,
        attack_type: int = 0,
        num_hacks: int = 0,
        hacked: bool = False,
        np_random: Optional[np.random.Generator] = None,
        num_attack_actions: int = 0,
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
        self.non_attacks = 2
        self.attack_successful = False
        self.stages = np.zeros((1, 4), dtype=int)
        self.stage_time_spent = np.zeros((1, 4), dtype=int)
        self.exfiltration_level: int = 0
        self.encryption_level: int = 0
        self.percent_benign_completed = np.zeros((1, 4), dtype=bool)
        self.local_detector_scores: np.ndarray = np.zeros((1, 4))
        self.global_detector_score: int = 0
        self.cross_layer_X = []

        self.attack_values: np.ndarray = (
            attack_values if attack_values is not None else np.zeros((1, 1))
        )
        self.defense_values: np.ndarray = (
            defense_values if defense_values is not None else np.zeros((1, 1))
        )
        self.defense_det: Optional[np.ndarray] = defense_det
        self.attacker_pos: Tuple[int, int] = attacker_pos
        self.game_step: int = game_step
        self.attacker_cumulative_reward: float = attacker_cumulative_reward
        self.defender_cumulative_reward: float = defender_cumulative_reward
        self.num_games: int = num_games
        self.attack_events: List[int] = (
            attack_events if attack_events is not None else []
        )
        self.attack_history: List[int] = list(self.attack_events)
        self.defense_events: List[AttackDefenseEvent] = (
            defense_events if defense_events is not None else []
        )
        self.defense_history: List[AttackDefenseEvent] = list(self.defense_events)
        self.done = done
        self.detected = detected
        self.attack_defense_type = attack_type
        self.num_hacks = num_hacks
        self.hacked = hacked
        self.np_random: np.random.Generator = (
            np_random if np_random is not None else np.random.default_rng()
        )
        self.action_descriptors = ["RE", "F1", "F2", "EX"]
        self.num_attack_actions = num_attack_actions

    def default_state(
        self,
        num_attack_types: int,
        # num_rows: int = 10,
        # num_cols: int = 10,
        randomize_state: bool = False,
        randomize_visibility: bool = False,
        visibility_p: float = 0.5,
    ) -> None:
        """
        Creates a default state

        :param num_attack_types: the number of attack types
        :param randomize_state: boolean flag whether to create the state randomly
        :param randomize_visibility: boolean flag whether to randomize visibility for partially observed envs
        :param visibility_p: probability of visibility
        :return: None
        """
        self.set_state(
            stages=np.zeros((1, 4)),
            exfiltration_level=0,
            encryption_level=0,
            percent_benign_completed=np.zeros((1, 4)),
            local_detector_scores=np.zeros((1, 4)),
            global_detector_score=0,
        )
        # self.defense_det = np.zeros((num_rows * num_cols, num_attack_types))
        # self.defense_values = np.zeros((num_rows * num_cols, num_attack_types))
        # self.attack_values = np.zeros((num_rows * num_cols, num_attack_types))
        self.attacker_pos = (0, 0)
        self.game_step = 0
        self.attacker_cumulative_reward = 0.0
        self.defender_cumulative_reward = 0.0
        self.num_games = 0
        self.attack_events = []
        self.attack_history = []
        self.defense_events = []
        self.defense_history = []
        self.done = False
        self.detected = False
        self.attack_defense_type = 0
        self.num_hacks = 0
        self.hacked = False
        self.attack_successful = False

    def set_state(
        self,
        stages: Optional[np.ndarray] = None,
        exfiltration_level: int = 0,
        encryption_level: int = 0,
        percent_benign_completed=None,
        local_detector_scores: Optional[np.ndarray] = None,
        global_detector_score: int = 0,
        num_attack_actions: int = 0,
    ):
        """
        Sets the state

        :param stages: stages array
        :param exfiltration_level: exfiltration progress bar level, 0..N_PROGRESS_STEPS
        :param encryption_level: encryption progress bar level, 0..N_PROGRESS_STEPS
        :param percent_benign_completed: percent benign completed as progress bar
        :param local_detector_scores: local detector scores
        :param global_detector_score: global detector score progress bar level, 0..N_PROGRESS_STEPS
        :param num_attack_actions: number of possible attack actions taken
        :return: None
        """
        self.stages = stages if stages is not None else np.zeros((1, 4))
        self.exfiltration_level = int(exfiltration_level)
        self.encryption_level = int(encryption_level)
        self.percent_benign_completed = (
            percent_benign_completed
            if percent_benign_completed is not None
            else np.zeros((1, 4), dtype=np.bool)
        )
        self.local_detector_scores = (
            local_detector_scores
            if local_detector_scores is not None
            else np.zeros((1, 4))
        )
        self.global_detector_score = int(global_detector_score)
        self.num_attack_actions = num_attack_actions

    def new_game(
        self,
        init_state: "GameState",
        a_reward: float = 0.0,
        d_reward: float = 0.0,
        update_stats: bool = True,
        randomize_state: bool = False,
        # num_attack_types: int = 0,
        np_random: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Updates the current state for a new game

        :param init_state: the initial state of the first game
        :param a_reward: the reward delta to increment or decrement the attacker cumulative reward with
        :param d_reward: the reward delta to increment or decrement the defender cumulative reward with
        :param update_stats: whether to update stats
        :param randomize_state: boolean flag whether to create the state randomly
        :param num_attack_types: number of attack types
        :param np_random: random number generator
        :return: None
        """
        if update_stats:
            self.num_games += 1
            if self.hacked or self.detected:
                self.attacker_cumulative_reward += a_reward
                self.defender_cumulative_reward += d_reward
                if self.hacked:
                    self.num_hacks += 1
        self.done = False
        self.attack_defense_type = 0
        self.game_step = 0
        self.attack_events = []
        self.attack_history = []
        self.defense_events = []
        self.defense_history = []
        self.stages = np.zeros((1, 4))
        self.stage_time_spent = np.zeros((1, 4), dtype=int)
        self.exfiltration_level = 0
        self.encryption_level = 0
        self.percent_benign_completed = np.zeros((1, 4), dtype=bool)
        self.local_detector_scores = np.zeros((1, 4))
        self.global_detector_score = 0
        if np_random is not None:
            self.np_random = np_random

        if not randomize_state:
            self.attack_values = np.copy(init_state.attack_values)
            self.defense_values = np.copy(init_state.defense_values)
        else:
            self.set_state()
        self.detected = False
        self.hacked = False
        self.attack_successful = False

    def copy(self) -> "GameState":
        """
        Creates a copy of the state

        :return: a copy of the current state
        """
        new_state = GameState()
        for attr in [
            "attack_values",
            "defense_values",
            "stages",
            "stage_time_spent",
            "percent_benign_completed",
            "local_detector_scores",
        ]:
            setattr(new_state, attr, np.copy(getattr(self, attr)))

        # defense_det is Optional, and np.copy(None) yields a 0-d object array rather
        # than None, so it cannot go through the loop above.
        new_state.defense_det = (
            None if self.defense_det is None else np.copy(self.defense_det)
        )

        for attr in [
            "exfiltration_level",
            "encryption_level",
            "global_detector_score",
            "attacker_pos",
            "game_step",
            "attacker_cumulative_reward",
            "defender_cumulative_reward",
            "num_games",
            "done",
            "detected",
            "attack_defense_type",
            "num_hacks",
            "hacked",
            "np_random",
            "attack_successful",
            # Set by set_state rather than by the constructor, and read by
            # simulate_stage to recognize the TERMINATE action, so a copy that dropped
            # it would silently stop honouring that action.
            "num_attack_actions",
        ]:
            setattr(new_state, attr, getattr(self, attr))

        new_state.attack_events = list(self.attack_events)
        new_state.attack_history = list(self.attack_history)
        new_state.defense_events = list(self.defense_events)
        new_state.defense_history = list(self.defense_history)
        return new_state

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
    def _calculate_exponential_probability(
        p_base: float, p_progress: float, n: int
    ) -> float:
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

    @classmethod
    def _advanced_progress_level(
        cls, level: int, value: float, trips: np.ndarray
    ) -> int:
        """
        Advances an ordinal progress bar to the level implied by `value`.

        The bar is monotone within an episode: `value` can fall (the consecutive-attempt
        run resets whenever the attacker switches stage) but the level never does.

        :param level: current level of the bar
        :param value: progress signal, compared against the trip points
        :param trips: strictly increasing trip points, one per level
        :return: the new level, in [0, len(trips)]
        """
        reached = int(np.searchsorted(trips, value, side="right"))
        return max(level, reached)

    def advance_global_detector_score(self, score: float) -> None:
        """
        Advances the global detector score progress bar to the level implied by
        `score`, using the same monotonic (never-decreasing) bar semantics as
        `exfiltration_level`/`encryption_level`.

        :param score: raw global detector score, in its theoretical range [0, 1]
        :return: None
        """
        self.global_detector_score = self._advanced_progress_level(
            self.global_detector_score, score, self.DETECTOR_SCORE_TRIPS
        )

    def get_consecutive_attack_attempts(self, attack_type: int) -> int:
        history_len = 10
        consecutive_attempts = 0
        attack_history = (
            self.attack_history[history_len:]
            if len(self.attack_history) > history_len
            else self.attack_history
        )

        for k, g in groupby(reversed(attack_history)):
            if k == attack_type:
                consecutive_attempts = len(list(g))
            break

        return consecutive_attempts

    def _get_consecutive_attack_requirement(self, attack_type: int) -> int:
        return self.CONSECUTIVE_ATTEMPT_REQUIREMENTS.get(attack_type, 2)

    def get_attack_probability(self, attack_type: int) -> float:
        p_base, p_progress = self.ATTACK_CONFIGS.get(attack_type, (0.1, 0.1))
        return self._calculate_exponential_probability(
            p_base, p_progress, self.get_consecutive_attack_attempts(attack_type)
        )

    # Behavior traces used as detector-scoring proxies for each attack stage. Keys not
    # present here (e.g. RECONNAISSANCE, IDLE, TERMINATE) have no detector sample.
    DETECTOR_STAGE_BEHAVIORS = MappingProxyType(
        {
            EXFILTRATION: "browser_compute",
            ENCRYPTION: "recon_mount",
        }
    )
    DETECTOR_DURATION_CHOICES = (1, 2, 3)

    def sample_detector_stage_lens(
        self, attack_type: int
    ) -> Optional[List[Tuple[str, int]]]:
        """
        Builds a synthetic (behavior, duration) sample describing the given attack
        stage, for scoring against the global lifecycle detector. Uses the state's own
        RNG so the sample stays reproducible from the episode seed.

        :param attack_type: the attack stage to build a sample for
        :return: a single-element (behavior, duration) list, or None if `attack_type`
            has no detector-behavior mapping
        """
        behavior = self.DETECTOR_STAGE_BEHAVIORS.get(attack_type)
        if behavior is None:
            return None
        duration = self.np_random.choice(self.DETECTOR_DURATION_CHOICES)
        return [(behavior, duration)]

    def simulate_stage(self, attack_type: int, exponential: bool = True) -> float:
        """
        Generic helper to simulate any attack type.
        """
        self.add_attack_event(attack_type)
        reward = self.DEFAULT_ATTACK_REWARD

        assert self.np_random is not None

        # no reward for benign, idle, or self-termination
        if attack_type < self.non_attacks:
            # if attacker terminates
            if attack_type == 0:
                self.done = True
            return reward

        # TODO translate attack into stage
        # penalize staying in completed stages
        if self.stages[0, attack_type] == 1:
            return reward

        # TODO translate attack into stage
        if attack_type < self.stage_time_spent.shape[1]:
            self.stage_time_spent[0, attack_type] = (
                self.get_consecutive_attack_attempts(attack_type)
            )

        reward += self.PROGRESS_REWARD

        if exponential:
            p = self.get_attack_probability(attack_type)
            if (
                attack_type == self.EXFILTRATION
                and self.stages[0, self.COMPRESSION] == 1
            ):
                p = np.clip(p * 2, 0, 1)

            # Update progress bars
            if attack_type == self.EXFILTRATION:
                self.exfiltration_level = self._advanced_progress_level(
                    self.exfiltration_level, p, self.PROGRESS_TRIPS
                )

            elif attack_type == self.ENCRYPTION:
                self.encryption_level = self._advanced_progress_level(
                    self.encryption_level, p, self.PROGRESS_TRIPS
                )

            stage_success = self.np_random.binomial(1, p) == 1

        else:
            requirement = self._get_consecutive_attack_requirement(attack_type)
            consecutive = self.get_consecutive_attack_attempts(attack_type)

            if (
                attack_type == self.EXFILTRATION
                and self.stages[0, self.COMPRESSION] == 1
            ):
                consecutive = np.clip(consecutive * 2, 0, requirement)

            # Update progress bars
            progress = consecutive / requirement
            if attack_type == self.EXFILTRATION:
                self.exfiltration_level = self._advanced_progress_level(
                    self.exfiltration_level, progress, self.CONSECUTIVE_PROGRESS_TRIPS
                )

            elif attack_type == self.ENCRYPTION:
                self.encryption_level = self._advanced_progress_level(
                    self.encryption_level, progress, self.CONSECUTIVE_PROGRESS_TRIPS
                )

            stage_success = consecutive >= requirement

        if stage_success:
            self.stages[0, attack_type] = 1
            reward += self.STAGE_REWARD

            if attack_type == self.EXFILTRATION:
                reward += self.EXFILTRATION_REWARD
                self.exfiltration_level = self.N_PROGRESS_STEPS

            if attack_type == self.ENCRYPTION:
                reward += self.ENCRYPTION_REWARD
                self.encryption_level = self.N_PROGRESS_STEPS
                self.attack_successful = True
                self.hacked = True
                self.done = True

        return reward

    def simulate_detection(self, attack_type: int) -> bool:
        """
        Implement this:

        :param attack_type: attack used that may be detected
        :param reconnaissance: boolean flag, if true simulate detection of reconnaissance activity
        :return: True if the node was detected, otherwise False
        """
        assert self.np_random is not None

        if attack_type >= self.stages.shape[1]:
            return False

        p = np.sum(self.stage_time_spent[0, attack_type]) / 100

        detected = self.np_random.binomial(1, p) == 1

        if detected:
            self.done = True
            self.detected = True

        return detected

    def defense_score(self, game_config):
        if not game_config.ransomware:
            return -1
        else:
            return 1.0 - self.encryption_level / self.N_PROGRESS_STEPS

    def get_attacker_observation(self) -> dict[str, int | np.ndarray]:
        """
        Converts the state of the dynamical system into an observation for the attacker. As the environment
        is a partially observed markov decision process, the attacker observation is only a subset of the game state

        :param local_view: boolean flag indicating whether observations are provided in a local view or not
        :return: An observation of the environment
        """
        attacker_observation = {
            "stages": self.stages.flatten().astype(int),
            "exfiltration_level": int(self.exfiltration_level),
            "encryption_level": int(self.encryption_level),
        }
        return attacker_observation

    def add_attack_event(self, attack_type: int) -> None:
        """
        Adds an attack event to the state

        :param attack_type: the type of the attack
        :return: None
        """
        self.attack_events.append(attack_type)
        self.attack_history.append(attack_type)

    def add_defense_event(self, target_pos: Tuple[int, int], defense_type: int) -> None:
        """
        Adds a defense event to the state

        :param target_pos: the position in the grid of the target node
        :param defense_type: the type of the defense
        :return: None
        """
        defense_event = AttackDefenseEvent(target_pos, defense_type)  # type: ignore
        self.attack_defense_type = defense_type
        self.defense_events.append(defense_event)
        self.defense_history.append(defense_event)

    def get_defender_observation(self) -> dict[str, int | np.ndarray]:
        """
        Converts the state of the dynamical system into an observation for the defender. As the environment
        is a partially observed markov decision process, the defender observation is only a subset of the game state

        :return: An observation of the environment
        """
        defender_observation = {
            # "stages": self.stages.flatten().astype(int),
            "encryption_level": int(self.encryption_level),
            # "local_detector_scores": self.local_detector_scores.astype(np.float32),
            "global_detector_score": int(self.global_detector_score),
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
        filehandler = open(path, "rb")
        return pickle.load(filehandler)

    @staticmethod
    def save(path, state):
        filehandler = open(path + "/initial_state.pkl", "wb")
        pickle.dump(state, filehandler)

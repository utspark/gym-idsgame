import csv
import os
import pickle
import time
from pathlib import Path

import gymnasium as gym
import joblib
import numpy as np

from typing import Union, Any, Literal, Optional, List, Tuple
from abc import ABC, abstractmethod

from numpy import ndarray

from detector_framework.cross_layer import cross_layer_train_run as cld
from detector_framework.global_detector import global_detector
from gym_idsgame.envs.constants import constants
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_ransomgame.envs.dao import ransomgame_config
from gym_ransomgame.envs.rendering.viewer import Viewer

from gym_ransomgame.agents.bot_agents.kill_chain_attacker_bot_agent import (
    KillChainAttackerBotAgent,
)
from gym_ransomgame.agents.bot_agents.threshold_defender_bot_agent import (
    ThresholdDefenderBotAgent,
)
from gym_ransomgame.envs.dao.game_config import GameConfig
from gym_ransomgame.envs.dao.game_state import GameState
from gym_ransomgame.envs.dao.ransomgame_config import RansomGameConfig
import gym_ransomgame.envs.util.ransomgame_util as util


class RansomGameEnv(gym.Env, ABC):
    """
    Implementation of the RL environment from the paper
    "Adversarial Reinforcement Learning in a Cyber Security Simulation" by Elderman et. al.

    It is an abstract cybersecurity simulation where an attacker agent tries to execute a ransomware lifecycle on a
    victim system while a defender agent attempts to defend the system.
    """

    def __init__(
        self,
        ransomgame_config: RansomGameConfig,
        save_dir: str,
        initial_state_path: str,
    ):
        """
        Initializes the environment

        Observation:
            Type: Box(num_nodes*num_attack_types)
        Actions:
            Type: Discrete(num_nodes*num_action_types)
        Reward:
            Reward is 0 for all steps except the final step which is either +100 (win) or -100 (loss)
        Starting State:
            Start node, all attack values are 0
        Episode Termination:
            When attacker reaches DATA node or when attacker is detected

        :param ransomgame_config: configuration of the environment
        :param save_dir: directory to save outputs, e.g. initial state
        :param initial_state_path: path to the initial state (if none, use default)
        """

        """
        if ransomgame_config is None:
            ransomgame_config = RansomGameConfig(
                render_config=RenderConfig(),
                game_config=GameConfig(),
                defender_agent=Agent(),
                attacker_agent=Agent(),
                initial_state_path=initial_state_path,
            )
        """
        self.save_dir = save_dir
        # self.validate_config(ransomgame_config)
        self.ransomgame_config: RansomGameConfig = ransomgame_config
        game_config = ransomgame_config.game_config

        if game_config.initial_state is None:
            raise ValueError("initial_state cannot be None")

        game_config.initial_state.set_state(
            # time=0,
            stages=np.zeros((1, 4)),
            exfiltration_level=0,
            encryption_level=0,
            percent_benign_completed=np.zeros((1, 4), dtype=bool),
            local_detector_scores=np.zeros((1, 4)),
            global_detector_score=0,
            num_attack_actions=game_config.num_attack_actions,
        )
        # Copy rather than alias: reset() passes game_config.initial_state to
        # state.new_game() as the template to restore from, so the two must be distinct
        # objects or the template is whatever the last episode left behind.
        self.state = game_config.initial_state.copy()

        is_attacker = game_config.attacker

        self.attacker_observation_space = game_config.get_attacker_observation_space()
        self.defender_observation_space = game_config.get_defender_observation_space()
        # reset() and step() always emit both observations, whichever agent is external,
        # so the declared space is the pair. Declaring only one agent's space made the
        # Gymnasium passive checker reject every observation. Use the per-agent
        # attacker_/defender_observation_space attributes to size a single-agent model.
        self.observation_space = gym.spaces.Tuple(
            (self.attacker_observation_space, self.defender_observation_space)
        )

        self.attacker_action_space = game_config.get_action_space(defender=False)
        self.defender_action_space = game_config.get_action_space(defender=True)
        self.action_space = (
            self.attacker_action_space if is_attacker else self.defender_action_space
        )

        self.viewer = None
        self.steps_beyond_done = None
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render.modes": ["human", "rgb_array"],
            "render_fps": 50,
            "video.frames_per_second": 50,  # Video rendering speed
        }
        self._gym_version = gym.__version__
        self.reward_range = (
            float(constants.GAME_CONFIG.NEGATIVE_REWARD),
            float(constants.GAME_CONFIG.POSITIVE_REWARD),
        )

        # Attacker state includes stage progress; defender only sees encryption progress.
        # TODO think about what the environment states are
        #  - they are different between attacker and defender
        #  - attack: what stages have been achieved/reached
        #  - defense: what alarms have triggered?
        #  - the detector scores and stage_time_spent are not observed yet
        self.num_attacker_states = int(np.prod(self.attacker_state_dims))
        self.num_defender_states = int(np.prod(self.defender_state_dims))
        self.num_states_full = (
            self.num_attacker_states if is_attacker else self.num_defender_states
        )

        self.num_attack_actions = game_config.num_attack_actions
        self.num_defense_actions = game_config.num_defense_actions
        self.past_moves = []
        self.past_positions = []
        self.hacked_nodes = []
        self.save_initial_state()
        self.a_cumulative_reward = 0
        self.d_cumulative_reward = 0
        self.game_trajectories = []
        self.game_trajectory = []
        self.attack_detections = []
        self.total_attacks = []
        self.defenses = []
        self.attacks = []
        self.num_failed_attacks = 0
        self.failed_attacks = {}
        MODEL_DIR = (
            Path(__file__).resolve().parents[1] / "detector_framework/data/models"
        )
        model_paths = {
            "syscall_clf_path": MODEL_DIR / "syscall_clf.joblib",
            "network_clf_path": MODEL_DIR / "network_clf.joblib",
            "hpc_clf_path": MODEL_DIR / "hpc_clf.joblib",
        }
        la_components = {
            "density": True,
            "propagation": True,
        }
        self.gd = global_detector.LifecycleDetector(
            **model_paths, lifecycle_awareness=True, stage_filter=False, **la_components
        )
        # Loaded once and held for the environment's lifetime rather than per-step,
        # since it's a static resource shared across all episodes/steps.
        FEATURE_FRAMES_PATH = (
            Path(__file__).resolve().parents[1]
            / "data/trace_data/feature_frames.joblib"
        )
        self.feature_frames = joblib.load(FEATURE_FRAMES_PATH)

    def _update_detector_score(self, attack_type: int) -> None:
        """
        Scores a synthetic trace window for the given attack stage against the global
        lifecycle detector and records the result on the state.

        :param attack_type: the attack stage just executed this step
        :return: None
        """
        WINDOW_SIZE_TIME = 0.7
        WINDOW_STRIDE_TIME = 0.1

        stage_lens = self.state.sample_detector_stage_lens(attack_type)
        if stage_lens is None:
            return

        tmp_cross_layer_X = cld.build_cross_layer_X(
            self.feature_frames,
            stage_lens,
            WINDOW_SIZE_TIME,
            WINDOW_STRIDE_TIME,
        )

        if len(self.state.cross_layer_X) > 0:
            self.state.cross_layer_X = tuple(
                np.concatenate((old, tmp), axis=0)
                for old, tmp in zip(self.state.cross_layer_X, tmp_cross_layer_X)
            )

        else:
            self.state.cross_layer_X = tmp_cross_layer_X

        self.state.advance_global_detector_score(
            self.gd.score_cross_layer(
                self.state.cross_layer_X
                # tmp_cross_layer_X
            )
        )

    # -------- API ------------
    def step(
        self, action: Any
    ) -> tuple[tuple[dict, dict], tuple[float, float], bool, bool, dict]:
        """
        Takes a step in the environment using the given action.

        When end of episode is reached, the caller is responsible for calling `reset()`
        to reset this environment's state.

        :param action: the action to take in the environment
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step()
            calls will return undefined results info (dict): contains auxiliary
            diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Initialization
        trajectory: List[Any] = [self.state]
        reward: Tuple[float, float] = (-0.1, float(0.1))
        info = {
            "detected": False,
            "ransomware": self.ransomgame_config.game_config.ransomware,
        }

        if self.state.game_step > constants.GAME_CONFIG.MAX_GAME_STEPS:
            obs = self.get_observation()
            return (
                obs,
                self._terminal_reward(
                    False, float(100 * constants.GAME_CONFIG.NEGATIVE_REWARD)
                ),
                True,
                False,
                info,
            )

        attack_action, defense_action = action

        # 1. Interpret attacker action. On a benign episode there is no ransomware
        # operator, so the env plays benign traffic in place of whoever would otherwise
        # act -- the external agent in a two-player game, or the attacker bot in a
        # defender-only one. Keeping the substitution here makes this the only place
        # benign behaviour is defined and leaves the attacker policies pure.
        if self.ransomgame_config.game_config.ransomware:
            attack_action = self.get_attacker_action(attack_action)
        else:
            attack_action = self.ransomgame_config.benign_agent.action(self.state)
        trajectory.append([attack_action])

        # 2. Interpret defense action
        defense_action = self.get_defender_action(defense_action)
        trajectory.append([defense_action])

        # 3. Defend
        # detect = defense_type == self.ransomgame_config.game_config.num_attack_types
        # defense_successful = self.state.defend(defense_type)
        # if defense_successful:
        #     self.defenses.append((defense_node_id, defense_type, detect, self.state.game_step))
        # self.state.add_defense_event(defense_pos, defense_type)

        # 4. Attack
        # self.state.attack(attack_type=attack_action)
        # self.state.add_attack_event(target_pos, attack_type, self.state.attacker_pos, reconnaissance)
        # self.attacks.append((target_node_id, attack_type, self.state.game_step, reconnaissance))

        attacker_reward = self.state.simulate_stage(
            attack_type=attack_action, exponential=True
        )
        reward = (attacker_reward, reward[1])

        self._update_detector_score(attack_action)

        if self.ransomgame_config.save_attack_stats:
            self.total_attacks.append([attack_action, self.state.attack_successful])

        # 5. Detection
        if not self.ransomgame_config.game_config.defender:
            detected = self.state.simulate_detection(attack_action)
        else:
            detected = defense_action

        if detected:
            info["detected"] = True
            self.state.done = True
            self.state.detected = True
            reward = self._terminal_reward(True)
        elif self.state.done:
            attacker_reward, _ = reward
            reward = self._terminal_reward(False, attacker_reward)

        if self.ransomgame_config.save_attack_stats:
            self.attack_detections.append([detected, self.state.stages])

        if self.state.done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                gym.logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- "
                    "any further steps are undefined behavior."
                )
                self.steps_beyond_done += 1
        self.state.game_step += 1
        # self.state.time += 1
        obs = self.get_observation()
        if self.viewer is not None:
            self.viewer.gameframe.set_state(self.state)

        self.a_cumulative_reward += reward[0]
        self.d_cumulative_reward += reward[1]
        trajectory.append(reward[0])
        trajectory.append(reward[1])
        trajectory.append(self.state)
        if self.ransomgame_config.save_trajectories:
            self.game_trajectories.append(trajectory)
        # return obs, float(reward[0]), self.state.done, False, info
        return obs, reward, self.state.done, False, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        update_stats: bool = False,
    ) -> tuple[tuple[dict, dict], dict]:
        """
        Resets the environment and returns the initial state

        Follows the Gymnasium reset contract, (observation, info), where the observation
        is itself the (attacker_obs, defender_obs) pair. Returning the two observations
        as the top-level tuple would put the defender observation in the info slot of
        every wrapper in the stack.

        :param seed: random seed
        :param options: options for resetting
        :param update_stats: whether the game count should be incremented or not
        :return: ((attacker_obs, defender_obs), info), where info["ransomware"] is the
                 episode type nature drew for the episode that is about to start
        """
        super().reset(seed=seed)
        self.action_space._np_random = self.np_random
        self.attacker_action_space._np_random = self.np_random
        self.defender_action_space._np_random = self.np_random
        if self.ransomgame_config.attacker_agent is not None:
            self.ransomgame_config.attacker_agent.np_random = self.np_random
        if self.ransomgame_config.defender_agent is not None:
            self.ransomgame_config.defender_agent.np_random = self.np_random

        # Nature's move. Drawing it here rather than in the training loop keeps the
        # episode type on the seeded env stream, makes _terminal_reward independent of
        # whoever is driving the env, and gives train and eval one shared definition.
        # A degenerate probability skips the draw so that an env pinned to one episode
        # type consumes no entropy and its episode stream is unaffected.
        p = self.ransomgame_config.ransomware_p
        if 0.0 < p < 1.0:
            self.ransomgame_config.game_config.ransomware = bool(
                self.np_random.random() < p
            )
        else:
            self.ransomgame_config.game_config.ransomware = p >= 1.0

        self.past_moves = []
        self.past_positions = []
        self.failed_attacks = {}
        self.steps_beyond_done = None
        initial_state = self.ransomgame_config.game_config.initial_state
        assert initial_state is not None
        self.state.new_game(
            initial_state,
            self.a_cumulative_reward,
            self.d_cumulative_reward,
            update_stats=update_stats,
            randomize_state=self.ransomgame_config.randomize_env,
            # num_attack_types=self.ransomgame_config.game_config.num_attack_types,
            np_random=self.np_random,
        )
        self.a_cumulative_reward = 0
        self.d_cumulative_reward = 0
        if self.viewer is not None:
            self.viewer.gameframe.reset()
        obs = self.get_observation()
        self.defenses = []
        self.attacks = []
        self.num_failed_attacks = 0
        info = {"ransomware": self.ransomgame_config.game_config.ransomware}
        return obs, info

    def restart(self) -> tuple[dict, dict]:
        """
        Restarts the game, and all the history

        :return: the (attacker_obs, defender_obs) observation from the first state
        """
        obs, _ = self.reset()
        self.state.restart()
        return obs

    def render(self, mode: str = "human"):
        """
        Renders the environment

        Supported rendering modes:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.

        :param mode: the rendering mode
        :return: True (if human mode) otherwise an rgb array
        """
        if mode not in self.metadata["render.modes"]:
            raise NotImplemented("mode: {} is not supported".format(mode))
        if self.viewer is None:
            self.__setup_viewer()
            self.viewer.gameframe.set_state(self.state)
        arr = self.viewer.render(return_rgb_array=mode == "rgb_array")
        self.state.attack_events = []
        self.state.defense_events = []
        return arr

    def close(self) -> None:
        """
        Closes the viewer (cleanup)

        :return: None
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            self.ransomgame_config.render_config.new_window()

    def save_initial_state(self) -> None:
        """
        Saves initial state to disk in binary npy format

        :return: None
        """
        if self.save_dir is not None and os.path.exists(self.save_dir):
            GameState.save(self.save_dir, self.state)

    def save_trajectories(self, checkpoint=True) -> None:
        """
        Saves the current list of game trajectories to disk

        :param checkpoint: boolean flag that indicates whether this is a checkpoint save or final save
        :return: None
        """
        suffix = ".pkl"
        if checkpoint:
            suffix = "_checkpoint.pkl"
        if self.ransomgame_config.save_trajectories:
            path = self.save_dir
            time_str = str(time.time())
            filehandler = open(path + "/trajectories_" + time_str + suffix, "wb")
            pickle.dump(self.game_trajectories, filehandler)
        else:
            self.game_trajectories = []

    def save_attack_data(self, checkpoint=True) -> None:
        """
        Saves the attack statistics to disk

        :param checkpoint: boolean flag that indicates whether this is a checkpoint save or final save
        :return: None
        """
        suffix = ".csv"
        if checkpoint:
            suffix = "_checkpoint.csv"
        if self.ransomgame_config.save_attack_stats:
            time_str = str(time.time())
            with open(
                self.save_dir + "/attack_detections_stats_" + time_str + suffix, "w"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["target_node", "detected", "detection_val"])
                for row in self.attack_detections:
                    writer.writerow(row)
            with open(self.save_dir + "/attack_stats_" + time_str + suffix, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["target_node", "attack_outcome"])
                for row in self.total_attacks:
                    writer.writerow(row)
        else:
            self.attack_detections = []
            self.total_attacks = []

    def _terminal_reward(
        self, detected: bool, attacker_reward: float = 0.0
    ) -> tuple[float, float]:
        """
        Compute terminal rewards for both agents.

        When detected:
            TP (ransomware=True):  attacker=-1, defender=defense_score (higher when caught early)
            FP (ransomware=False): attacker=+1, defender=-2
        When not detected:
            TN (ransomware=False): attacker=attacker_reward, defender=+1
            FN (ransomware=True):  attacker=attacker_reward, defender=-1

        :param detected: whether the defender raised an alarm
        :param attacker_reward: attacker reward to use when not detected (ignored when detected)
        :return: (attacker_reward, defender_reward)
        """
        ransomware = self.ransomgame_config.game_config.ransomware
        if detected:
            if ransomware:
                return float(-1), self.state.defense_score(
                    self.ransomgame_config.game_config
                )
            else:
                return float(1), float(-10)
        else:
            return attacker_reward, (1.0 if not ransomware else -1.0)

    # def get_successful_attack_reward(self, attack_action) -> tuple[Any, float]:
    #     """
    #     Returns the reward for the attacker and defender after a successful attack on some server in
    #     the network
    #
    #     :return:(attacker_reward, defender_reward)
    #     """
    #     return self.state.attack_score(self.ransomgame_config.game_config, attack_action), float(0)

    def get_observation(self) -> tuple[dict, dict]:
        """
        Returns an observation of the state

        :return: (attacker_obs, defender_obs)
        """
        attacker_obs = self.state.get_attacker_observation()
        defender_obs = self.state.get_defender_observation()

        return attacker_obs, defender_obs

    def local_view_features(self) -> bool:
        """
        Boolean function to check whether the environment uses local view observations of the attacker

        :return: True if the environment uses local view observations
        """
        return self.ransomgame_config.local_view_observations

    @abstractmethod
    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        pass

    @abstractmethod
    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        pass

    # -------- Private methods ------------

    def __setup_viewer(self):
        """
        Setup for the viewer to use for rendering
        :return: None
        """
        resource_path = (
            Path(__file__).parent / "rendering" / constants.RENDERING.RESOURCES_DIR
        )
        self.ransomgame_config.render_config.resources_dir = str(resource_path)
        self.viewer = Viewer(ransomgame_config=self.ransomgame_config)
        self.viewer.agent_start()

    @property
    def attacker_state_dims(self) -> Tuple[int, ...]:
        """
        Cardinality of each element of the attacker observation, in key order.

        The progress bars are ordinal (0..N_PROGRESS_STEPS) rather than 4 independent bits,
        so the radix is mixed: 2**4 stage combinations x 5 exfiltration levels x 5 encryption
        levels = 400 states, versus 2**12 = 4096 for the thermometer-coded equivalent.

        :return: the per-element cardinalities
        """
        n_levels = GameState.N_PROGRESS_STEPS + 1
        n_stages = self.ransomgame_config.game_config.stages.shape[1]
        return (2,) * n_stages + (n_levels, n_levels)

    @property
    def defender_state_dims(self) -> Tuple[int, ...]:
        """
        Cardinality of each element of the defender observation, in key order.

        :return: the per-element cardinalities
        """
        return (GameState.N_PROGRESS_STEPS + 1,)

    def get_state_id(self, observation: Any) -> int:
        """
        Convert a RansomGame observation into a stable integer state id.

        The id is a mixed-radix encoding of the observation, so it is a bijection onto
        [0, num_*_states) with no lookup table and no dynamic state discovery. Dispatches on
        the observation contents so that attacker and defender ids can be requested from the
        same env instance.

        :param observation: an attacker or defender observation
        :return: the state id
        """
        if "stages" in observation:
            key = tuple(int(x) for x in observation["stages"]) + (
                int(observation["exfiltration_level"]),
                int(observation["encryption_level"]),
            )
            dims = self.attacker_state_dims
        else:
            key = (int(observation["encryption_level"]),)
            dims = self.defender_state_dims

        return int(np.ravel_multi_index(key, dims))


class AttackerEnv(RansomGameEnv, ABC):
    """
    Abstract AttackerEnv of the RansomGameEnv.

    Environments where the defender is part of the environment and the environment is designed to be used by an
    attacker-agent should inherit this class
    """

    def __init__(
        self,
        ransomgame_config: RansomGameConfig,
        save_dir: str,
        initial_state_path: str,
    ):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param ransomgame_config: configuration of the environment (if not specified a default config is used)
        """
        if ransomgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        if ransomgame_config.defender_agent is None:
            raise ValueError("Cannot instantiate attacker-env without a defender agent")
        super().__init__(
            ransomgame_config=ransomgame_config,
            save_dir=save_dir,
            initial_state_path=initial_state_path,
        )

    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        """
        The attacker is external, so its action is whatever the caller passed.

        :param action: the attacker action supplied to step()
        :return: the attacker action
        """
        attacker_action = action
        return attacker_action

    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        """
        The defender is part of the environment, so the action supplied to step() is
        discarded and the defender bot is consulted instead.

        :param action: the defender action supplied to step() (ignored)
        :return: the defender bot's action
        """
        return self.ransomgame_config.defender_agent.action(self.state)


class DefenderEnv(RansomGameEnv, ABC):
    """
    Abstract DefenderEnv of the RansomGameEnv.

    Environments where the defender is part of the environment and the environment is designed to be used by an
    defender-agent should inherit this class
    """

    def __init__(
        self,
        ransomgame_config: RansomGameConfig,
        save_dir: str,
        initial_state_path: str,
    ):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param ransomgame_config: configuration of the environment (if not specified a default config is used)
        """
        if ransomgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        if ransomgame_config.attacker_agent is None:
            raise ValueError(
                "Cannot instantiate defender-env without an attacker agent"
            )
        super().__init__(
            ransomgame_config=ransomgame_config,
            save_dir=save_dir,
            initial_state_path=initial_state_path,
        )

    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        """
        The attacker is part of the environment, so the action supplied to step() is
        discarded and the attacker bot is consulted instead.

        :param action: the attacker action supplied to step() (ignored)
        :return: the attacker bot's action
        """
        return self.ransomgame_config.attacker_agent.action(self.state)

    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        """
        The defender is external, so its action is whatever the caller passed.

        :param action: the defender action supplied to step()
        :return: the defender action
        """
        defender_action = action
        return defender_action


class AttackDefenseEnv(RansomGameEnv, ABC):
    """
    Abstract AttackDefenseEnv of the RansomGameEnv.

    Environments where both the attacker and defender are external to the environment should inherit this class.
    """

    def __init__(
        self,
        ransomgame_config: RansomGameConfig,
        save_dir: str,
        initial_state_path: str,
    ):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param ransomgame_config: configuration of the environment (if not specified a default config is used)
        """
        if ransomgame_config is None:
            raise ValueError("Cannot instantiate env without configuration")
        super().__init__(
            ransomgame_config=ransomgame_config,
            save_dir=save_dir,
            initial_state_path=initial_state_path,
        )

    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        """
        Both agents are external, so the action is whatever the caller passed.

        :param action: the defender action supplied to step()
        :return: the defender action
        """
        defender_action = action
        return defender_action

    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        """
        Both agents are external, so the action is whatever the caller passed.

        :param action: the attacker action supplied to step()
        :return: the attacker action
        """
        attacker_action = action
        return attacker_action


class RansomGameMinimalDefenseV0Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 0
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    """

    def __init__(
        self,
        ransomgame_config: RansomGameConfig,
        save_dir: str,
        initial_state_path: str,
    ):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param ransomgame_config: configuration of the environment (if not specified a default config is used)
        """
        if ransomgame_config is None:
            game_config = GameConfig(
                manual_attacker=False,
                manual_defender=False,
                attacker=True,
                defender=False,
                num_attack_types=2,
                initial_state_path=None,
                ransomware=True,
            )
            game_config.set_initial_state(defense_val=2, attack_val=0)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = ThresholdDefenderBotAgent(game_config)
            ransomgame_config = RansomGameConfig(
                game_config=game_config,
                defender_agent=defender_agent,
                initial_state_path=None,
                render_config=RenderConfig(),
                # Every episode is a ransomware episode: there is no benign traffic for
                # the attacker to hide in, and nothing here rewards it for abstaining.
                ransomware_p=1.0,
            )
            ransomgame_config.render_config.caption = "ransomgame-minimal_defense-v0"
        super().__init__(
            ransomgame_config=ransomgame_config,
            save_dir=save_dir,
            initial_state_path=None,
        )


class RansomGameMinimalAttackV0Env(DefenderEnv):

    def __init__(
        self,
        ransomgame_config: RansomGameConfig,
        save_dir: str,
        initial_state_path: str,
    ):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param ransomgame_config: configuration of the environment (if not specified a default config is used)
        """
        if ransomgame_config is None:
            game_config = GameConfig(
                manual_attacker=False,
                manual_defender=False,
                attacker=False,
                defender=True,
                num_attack_types=2,
                initial_state_path=None,
                ransomware=True,
            )
            game_config.set_initial_state(defense_val=2, attack_val=0)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            attacker_agent = KillChainAttackerBotAgent(game_config)
            ransomgame_config = RansomGameConfig(
                game_config=game_config,
                attacker_agent=attacker_agent,
                initial_state_path=None,
                render_config=RenderConfig(),
                # Half the episodes are benign, so the defender faces a real
                # signal-detection problem rather than a dominant alarm-immediately
                # policy. The bot attacker idles for the whole of a benign episode.
                ransomware_p=0.5,
            )
            ransomgame_config.render_config.caption = "ransomgame-minimal_attack-v0"
        super().__init__(
            ransomgame_config=ransomgame_config,
            save_dir=save_dir,
            initial_state_path=None,
        )


class RansomGameV0Env(AttackDefenseEnv):
    """
    [Rewards] Sparse
    [Version] 0
    [Observations] partially observed
    [Environment] Deterministic
    """

    def __init__(
        self,
        ransomgame_config: RansomGameConfig,
        save_dir: str,
        initial_state_path: str,
    ):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param idsgame_config: configuration of the environment (if not specified a default config is used)
        """
        if ransomgame_config is None:
            game_config = GameConfig(
                manual_attacker=False,
                manual_defender=False,
                attacker=True,
                defender=True,
                num_attack_types=2,
                initial_state_path=None,
                ransomware=True,
            )
            game_config.set_initial_state(defense_val=2, attack_val=0)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            ransomgame_config = RansomGameConfig(
                game_config=game_config,
                initial_state_path=None,
                render_config=RenderConfig(),
                # Half the episodes are benign, on which the env plays benign_agent
                # instead of the external attacker. Without them the defender faces no
                # false-positive risk and alarming on the first step strictly dominates.
                ransomware_p=0.5,
            )
            ransomgame_config.render_config.caption = "ransomgame-v0"
        super().__init__(
            ransomgame_config=ransomgame_config,
            save_dir=save_dir,
            initial_state_path=initial_state_path,
        )

import csv
import itertools
import os
import pickle
import time
from pathlib import Path

import gymnasium as gym
import math
import numpy as np

from typing import Union, Any, Literal, Optional, List, Tuple
from abc import ABC, abstractmethod

from numpy import ndarray

from gym_idsgame.envs.constants import constants
from gym_idsgame.agents.agent import Agent
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_ransomgame.envs.rendering.viewer import Viewer

from gym_idsgame.agents.bot_agents.bot_agent import BotAgent

class DummyAgent(BotAgent):

    def __init__(self, game_config):
        super(DummyAgent, self).__init__(game_config)

    def action(self, state):
        return 0

from gym_ransomgame.envs.dao.game_config import GameConfig
from gym_ransomgame.envs.dao.game_state import GameState
from gym_ransomgame.envs.dao.ransomgame_config import RansomGameConfig
from gym_idsgame.agents.bot_agents.defend_minimal_value_bot_agent import DefendMinimalValueBotAgent
import gym_ransomgame.envs.util.ransomgame_util as util


class RansomGameEnv(gym.Env, ABC):
    """
    Implementation of the RL environment from the paper
    "Adversarial Reinforcement Learning in a Cyber Security Simulation" by Elderman et. al.

    It is an abstract cybersecurity simulation where an attacker agent tries to execute a ransomware lifecycle on a
    victim system while a defender agent attempts to defend the system.
    """

    def __init__(self, ransomgame_config: RansomGameConfig, save_dir: str, initial_state_path: str):
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
        if self.ransomgame_config.game_config.initial_state is None:
            raise ValueError("initial_state cannot be None")
        self.state: GameState = self.ransomgame_config.game_config.initial_state
        self.observation_space = self.ransomgame_config.game_config.get_attacker_observation_space()
        self.action_space = self.ransomgame_config.game_config.get_action_space(defender=False)
        self.attacker_action_space = self.ransomgame_config.game_config.get_action_space(defender=False)
        self.defender_action_space = self.ransomgame_config.game_config.get_action_space(defender=True)
        self.viewer = None
        self.steps_beyond_done = None
        self.metadata = {
         'render_modes': ['human', 'rgb_array'],
         'render.modes': ['human', 'rgb_array'],
         'render_fps': 50,
         'video.frames_per_second' : 50 # Video rendering speed
        }
        self._gym_version = gym.__version__
        self.reward_range = (float(constants.GAME_CONFIG.NEGATIVE_REWARD), float(constants.GAME_CONFIG.POSITIVE_REWARD))

        self.n_state_elems = 14
        self.num_states = self.n_state_elems
        self.num_states_full = int(math.pow(self.ransomgame_config.game_config.max_value + 1, self.n_state_elems))

        self.num_attack_actions = self.ransomgame_config.game_config.num_attack_actions
        self.num_defense_actions = self.ransomgame_config.game_config.num_defense_actions
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

    # -------- API ------------
    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        """
        Takes a step in the environment using the given action.

        When end of episode is reached, the caller is responsible for calling `reset()`
        to reset this environment's state.

        :param action: the action to take in the environment
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Initialization
        trajectory: List[Any] = [self.state]
        reward: Tuple[float, float] = (-0.1, float(0))
        info = {"detected": False}

        if self.state.game_step > constants.GAME_CONFIG.MAX_GAME_STEPS:
            obs, _ = self.get_observation()
            return obs, float(100*constants.GAME_CONFIG.NEGATIVE_REWARD), True, False, info

        attack_action, defense_action = action

        # 1. Interpret attacker action
        if attack_action != -1:
            attack_action = self.get_attacker_action(attack_action)
            trajectory.append([attack_action])

        # 2. Interpret defense action
        defense_node_id, defense_pos, defense_type,  = self.get_defender_action(action)
        trajectory.append([defense_node_id, defense_pos, defense_type])

        # 3. Defend
        # detect = defense_type == self.ransomgame_config.game_config.num_attack_types
        # defense_successful = self.state.defend(defense_type)
        # if defense_successful:
        #     self.defenses.append((defense_node_id, defense_type, detect, self.state.game_step))
        # self.state.add_defense_event(defense_pos, defense_type)

        # 4. Attack
        if attack_action != -1:
            # self.state.attack(attack_type=attack_action)
            # self.state.add_attack_event(target_pos, attack_type, self.state.attacker_pos, reconnaissance)
            # self.attacks.append((target_node_id, attack_type, self.state.game_step, reconnaissance))

            reward = self.state.simulate_stage(attack_type=attack_action, exponential=True)

            if self.ransomgame_config.save_attack_stats:
                self.total_attacks.append([attack_action, self.state.attack_successful])

        else:
            #print("illegal action:{}".format(attack_action))
            reward = -1*constants.GAME_CONFIG.POSITIVE_REWARD, 0
            # self.state.done = True
            # self.state.detected = True

        # 5. Detection
        detected = self.state.simulate_detection(attack_action)
        if detected:
            info["detected"] = True
            reward = self.get_detect_reward()

        if self.ransomgame_config.save_attack_stats:
            self.attack_detections.append([detected, self.state.stages, self.state.stage_time_spent])

        if self.state.done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                gym.logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- "
                    "any further steps are undefined behavior.")
                self.steps_beyond_done += 1
        self.state.game_step += 1
        self.state.time += 1
        obs, _ = self.get_observation()
        if self.viewer is not None:
            self.viewer.gameframe.set_state(self.state)

        self.a_cumulative_reward += reward[0]
        self.d_cumulative_reward += reward[1]
        trajectory.append(reward[0])
        trajectory.append(reward[1])
        trajectory.append(self.state)
        if self.ransomgame_config.save_trajectories:
            self.game_trajectories.append(trajectory)
        return obs, float(reward[0]), self.state.done, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, update_stats: bool = False) -> tuple[dict, dict]:
        """
        Resets the environment and returns the initial state

        :param seed: random seed
        :param options: options for resetting
        :param update_stats: whether the game count should be incremented or not
        :return: the initial state
        """
        super().reset(seed=seed)
        self.action_space._np_random = self.np_random
        self.attacker_action_space._np_random = self.np_random
        self.defender_action_space._np_random = self.np_random
        if self.ransomgame_config.attacker_agent is not None:
            self.ransomgame_config.attacker_agent.np_random = self.np_random
        if self.ransomgame_config.defender_agent is not None:
            self.ransomgame_config.defender_agent.np_random = self.np_random
        self.past_moves = []
        self.past_positions = []
        self.hacked_nodes = []
        self.failed_attacks = {}
        self.steps_beyond_done = None
        initial_state = self.ransomgame_config.game_config.initial_state
        assert initial_state is not None
        self.state.new_game(initial_state, self.a_cumulative_reward,
                            self.d_cumulative_reward, update_stats=update_stats,
                            randomize_state=self.ransomgame_config.randomize_env,
                            num_attack_types=self.ransomgame_config.game_config.num_attack_types,
                            np_random=self.np_random)
        self.a_cumulative_reward = 0
        self.d_cumulative_reward = 0
        if self.viewer is not None:
            self.viewer.gameframe.reset()
        obs, _ = self.get_observation()
        self.defenses = []
        self.attacks = []
        self.num_failed_attacks = 0
        return obs, {}

    def restart(self) -> dict:
        """
        Restarts the game, and all the history

        :return: the observation from the first state
        """
        obs, info = self.reset()
        self.state.restart()
        return obs

    def render(self, mode: str ='human'):
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
        arr = self.viewer.render(return_rgb_array = mode=='rgb_array')
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

    def save_trajectories(self, checkpoint = True) -> None:
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
            filehandler = open(path + "/trajectories_" + time_str + suffix, 'wb')
            pickle.dump(self.game_trajectories, filehandler)
        else:
            self.game_trajectories = []

    def save_attack_data(self, checkpoint = True) -> None:
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
            with open(self.save_dir + "/attack_detections_stats_" + time_str + suffix, "w") as f:
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

    def get_detect_reward(self, *args, **kwargs) -> tuple[float, Any]:
        """
        Returns the attacker and defender reward in the case when the attacker was detected.

        :return: (attacker_reward, defender_reward)
        """
        return float(-1), self.state.defense_score(self.ransomgame_config.game_config)

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
        resource_path = Path(__file__).parent / "rendering" / constants.RENDERING.RESOURCES_DIR
        self.ransomgame_config.render_config.resources_dir = str(resource_path)
        self.viewer = Viewer(ransomgame_config=self.ransomgame_config)
        self.viewer.agent_start()

    def build_state_to_idx_map(self):
        # TODO think about what the environment states are
        #  - they are different between attacker and defender
        #  - attack: what stages have been achieved/reached
        #  - defense: what alarms have triggered?
        """
        Builds a map that maps states to index (useful when constructing Q-tables for example)

        :return: the lookup map
        """

        n_state_elems = len(self.ransomgame_config.game_config.stages)
        # n_state_elems += len(self.state.stage_time_spent)
        # n_state_elems += len(self.state.percent_exfiltrated)
        """
        n_state_elems += 1  # percent_exfiltrated
        n_state_elems += 1  # percent_encrypted
        n_state_elems += 3  # local_detector_score
        n_state_elems += 1  # global_detector_score
        """

        states = list(
            itertools.product(list(range(self.ransomgame_config.game_config.max_value + 1)), repeat=n_state_elems))
        assert int(len(states)) == int(math.pow(self.ransomgame_config.game_config.max_value + 1, n_state_elems))

        state_to_idx = {}
        for idx, s in enumerate(states):
            state_to_idx[s] = idx
        return state_to_idx


class AttackerEnv(RansomGameEnv, ABC):
    """
    Abstract AttackerEnv of the RansomGameEnv.

    Environments where the defender is part of the environment and the environment is designed to be used by an
    attacker-agent should inherit this class
    """

    def __init__(self, ransomgame_config: RansomGameConfig, save_dir: str, initial_state_path: str):
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
        super().__init__(ransomgame_config=ransomgame_config, save_dir=save_dir, initial_state_path=initial_state_path)
        self.observation_space = self.ransomgame_config.game_config.get_attacker_observation_space()

    def get_attacker_action(self, action) -> Union[int, Union[int, int], int]:
        attacker_action = action
        return attacker_action

    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        _, defender_action = action
        return 0, (0, 0), defender_action

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
    def __init__(self, ransomgame_config: RansomGameConfig, save_dir: str, initial_state_path: str):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param ransomgame_config: configuration of the environment (if not specified a default config is used)
        """
        if ransomgame_config is None:
            game_config = GameConfig(manual_attacker=False, num_attack_types=2, max_value=10, manual_defender=False,
                                     initial_state_path=None, ransomware=True)
            game_config.set_initial_state(defense_val=2, attack_val=0)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DummyAgent(game_config)
            ransomgame_config = RansomGameConfig(game_config=game_config, defender_agent=defender_agent,
                                                 initial_state_path=None, render_config=RenderConfig())
            ransomgame_config.render_config.caption = "ransomgame-minimal_defense-v0"
        super().__init__(ransomgame_config=ransomgame_config, save_dir=save_dir, initial_state_path=None)


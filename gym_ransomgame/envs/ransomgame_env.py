import gymnasium as gym
import math
import numpy as np

from typing import Union
from abc import ABC, abstractmethod
from gym_idsgame.envs.constants import constants
from gym_ransomgame.envs.dao.game_config import GameConfig
from gym_ransomgame.envs.dao.game_state import GameState


class RansomGameEnv(gym.Env, ABC):
    """
    Implementation of the RL environment from the paper
    "Adversarial Reinforcement Learning in a Cyber Security Simulation" by Elderman et. al.

    It is an abstract cybersecurity simulation where an attacker agent tries to execute a ransomware lifecycle on a
    victim system while a defender agent attempts to defend the system.
    """

    def __init__(self, ransomgame_config: GameConfig = None, save_dir: str = None, initial_state_path: str = None):
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
        if ransomgame_config is None:
            ransomgame_config = GameConfig(initial_state_path=initial_state_path)
        self.save_dir = save_dir
        self.validate_config(ransomgame_config)
        self.ransomgame_config: GameConfig = ransomgame_config
        self.state: GameState = self.ransomgame_config.game_config.initial_state.copy()
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
        import gymnasium as gym
        self._gym_version = gym.__version__
        self.reward_range = (float(constants.GAME_CONFIG.NEGATIVE_REWARD), float(constants.GAME_CONFIG.POSITIVE_REWARD))
        self.num_states = self.ransomgame_config.game_config.num_nodes
        self.num_states_full = int(math.pow(self.ransomgame_config.game_config.max_value + 1,
                                            self.ransomgame_config.game_config.num_nodes *
                                            (self.ransomgame_config.game_config.num_attack_types + 1)))
        if self.ransomgame_config.game_config.network_config.fully_observed:
            self.num_states_full = int(math.pow(self.ransomgame_config.game_config.max_value + 1,
                                                self.ransomgame_config.game_config.num_nodes *
                                                (self.ransomgame_config.game_config.num_attack_types + 1) * 2))
        self.num_attack_actions = self.ransomgame_config.game_config.num_attack_actions
        self.num_defense_actions = self.ransomgame_config.game_config.num_defense_actions
        self.past_moves = []
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
    def step(self, action: int) -> Union[np.ndarray, int, bool, dict]:
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
        import gym_idsgame.envs.util.idsgame_util as util
        # Initialization
        trajectory = []
        trajectory.append(self.state)
        reward = (0,0)
        info = {}
        info["moved"] = False
        self.state.attack_events = []
        self.state.defense_events = []

        if self.state.game_step > constants.GAME_CONFIG.MAX_GAME_STEPS:
            return self.get_observation()[0], (100*constants.GAME_CONFIG.NEGATIVE_REWARD,
                                            100*constants.GAME_CONFIG.NEGATIVE_REWARD), True, info

        attack_action, defense_action = action

        # 1. Interpret attacker action
        attacker_pos = self.state.attacker_pos
        if attack_action != -1:
            target_node_id, target_pos, attack_type, reconnaissance = self.get_attacker_action(action)
            trajectory.append([target_node_id, target_pos, attack_type, reconnaissance])

        # 2. Interpret defense action
        defense_node_id, defense_pos, defense_type,  = self.get_defender_action(action)
        trajectory.append([defense_node_id, defense_pos, defense_type])

        # 3. Defend
        detect = defense_type == self.ransomgame_config.game_config.num_attack_types
        defense_successful = self.state.defend(defense_node_id, defense_type, self.ransomgame_config.game_config.max_value,
                                               self.ransomgame_config.game_config.network_config, detect=detect)
        if defense_successful:
            self.defenses.append((defense_node_id, defense_type, detect, self.state.game_step))
        self.state.add_defense_event(defense_pos, defense_type)

        if attack_action != -1 and util.is_attack_legal(target_pos, attacker_pos, self.ransomgame_config.game_config.network_config,
                                past_positions=self.past_positions):
            self.past_moves.append(target_node_id)
            if not reconnaissance:
                # 4. Attack
                self.state.attack(target_node_id, attack_type, self.ransomgame_config.game_config.max_value,
                                  self.ransomgame_config.game_config.network_config,
                                  reconnaissance_enabled=self.ransomgame_config.reconnaissance_actions)
            else:
                rec_reward = self.state.reconnaissance(target_node_id, attack_type, reconnaissance_reward=self.ransomgame_config.reconnaissance_reward)
                self.past_reconnaissance_activities.append((target_node_id, attack_type))
                reward = (rec_reward, 0)

            self.state.add_attack_event(target_pos, attack_type, self.state.attacker_pos, reconnaissance)
            self.attacks.append((target_node_id, attack_type, self.state.game_step, reconnaissance))

            attack_successful = False
            if not reconnaissance:
                # 5. Simulate attack outcome
                attack_successful = self.state.simulate_attack(target_node_id, attack_type,
                                                               self.ransomgame_config.game_config.network_config)
            if self.ransomgame_config.save_attack_stats:
                self.total_attacks.append([target_node_id, attack_successful, reconnaissance])

            # 6. Update state based on attack outcome
            if attack_successful:
                if not reconnaissance:
                    info["moved"] = True
                    self.past_positions.append(target_pos)
                    self.state.attacker_pos = target_pos
                    self.hacked_nodes.append(target_node_id)
                    if target_pos == self.ransomgame_config.game_config.network_config.data_pos:
                        self.state.done = True
                        self.state.hacked = True
                        reward = self.get_hack_reward(attack_type, target_node_id)
                    else:
                        reward = self.get_successful_attack_reward(attack_type, target_node_id)
                self.num_failed_attacks = 0
            else:
                self.num_failed_attacks += 1
                if (target_node_id, attack_type) in self.failed_attacks:
                    self.failed_attacks[(target_node_id, attack_type)] = self.failed_attacks[(target_node_id, attack_type)] + 1
                else:
                    self.failed_attacks[(target_node_id, attack_type)] = 1
                self.past_positions.append(self.state.attacker_pos)
                detected = self.state.simulate_detection(target_node_id, reconnaissance=reconnaissance,
                                                         reconnaissance_detection_factor=self.ransomgame_config.reconnaissance_detection_factor,
                                                         np_random=self.np_random)
                if detected:
                    self.state.done = True
                    self.state.detected = True
                    reward = self.get_detect_reward(target_node_id,  attack_type, self.state.defense_det[target_node_id], reconnaissance)
                # else:
                #     if not reconnaissance:
                #         reward = self.get_blocked_attack_reward(target_node_id, attack_type)
                if self.ransomgame_config.save_attack_stats:
                    self.attack_detections.append([target_node_id, detected, self.state.defense_det[target_node_id]])
        else:
            #print("illegal action:{}".format(attack_action))
            reward = -1*constants.GAME_CONFIG.POSITIVE_REWARD, 0
            # self.state.done = True
            # self.state.detected = True

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
        observation = self.get_observation()
        if self.viewer is not None:
            self.viewer.gameframe.set_state(self.state)
        self.a_cumulative_reward += reward[0]
        self.d_cumulative_reward += reward[1]
        trajectory.append(reward[0])
        trajectory.append(reward[1])
        trajectory.append(self.state)
        if self.ransomgame_config.save_trajectories:
            self.game_trajectories.append(trajectory)
        return observation, reward, self.state.done, False, info

    def reset(self, seed: int = None, options: dict = None, update_stats = False) -> np.ndarray:
        """
        Resets the environment and returns the initial state

        :param seed: random seed
        :param options: options for resetting
        :param update_stats: whether the game count should be incremented or not
        :return: the initial state
        """
        if seed is None:
            seed = 0
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.attacker_action_space.seed(seed)
        self.defender_action_space.seed(seed)
        if self.ransomgame_config.attacker_agent is not None:
            self.ransomgame_config.attacker_agent.np_random = self.np_random
        if self.ransomgame_config.defender_agent is not None:
            self.ransomgame_config.defender_agent.np_random = self.np_random
        self.past_moves = []
        self.past_positions = []
        self.past_reconnaissance_activities = []
        self.failed_attacks = {}
        self.furthest_hack = self.ransomgame_config.game_config.network_config.num_rows - 1
        self.steps_beyond_done = None
        self.state.new_game(self.ransomgame_config.game_config.initial_state, self.a_cumulative_reward,
                            self.d_cumulative_reward, update_stats=update_stats,
                            randomize_state=self.ransomgame_config.randomize_env,
                            network_config=self.ransomgame_config.game_config.network_config,
                            num_attack_types=self.ransomgame_config.game_config.num_attack_types,
                            defense_val = self.ransomgame_config.game_config.defense_val,
                            attack_val = self.ransomgame_config.game_config.attack_val,
                            det_val = self.ransomgame_config.game_config.det_val,
                            vulnerability_val = self.ransomgame_config.game_config.vulnerabilitiy_val,
                            num_vulnerabilities_per_layer=self.ransomgame_config.game_config.num_vulnerabilities_per_node,
                            num_vulnerabilities_per_node=self.ransomgame_config.game_config.num_vulnerabilities_per_node,
                            randomize_visibility=self.ransomgame_config.randomize_visibility,
                            visibility_p=self.ransomgame_config.visibility_p,
                            np_random=self.np_random)
        self.a_cumulative_reward = 0
        self.d_cumulative_reward = 0
        if self.ransomgame_config.randomize_starting_position:
            self.state.randomize_attacker_position(self.ransomgame_config.game_config.network_config, np_random=self.np_random)
        if self.viewer is not None:
            self.viewer.gameframe.reset()
        observation = self.get_observation()
        self.past_positions.append(self.state.attacker_pos)
        self.defenses = []
        self.attacks = []
        self.hacked_nodes = []
        self.num_failed_attacks = 0
        return observation, {}

    def restart(self) -> np.ndarray:
        """
        Restarts the game, and all the history

        :return: the observation from the first state
        """
        obs = self.reset()
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

    def get_attacker_node_from_observation(self, observation: np.ndarray) -> int:
        """
        Extracts which node the attacker is currently at from the observation representation

        :param observation: the observation representation emitted from the environment
        :return: the id of the node that the attacker is in
        """
        return self.state.get_attacker_node_from_observation(
            observation, reconnaissance=self.ransomgame_config.game_config.reconnaissance_actions)

    def hack_probability(self) -> float:
        """
        :return: the cumulative hack-probabiltiy according to the game history
        """
        hack_probability = 0.0
        if self.state.num_hacks > 0:
            hack_probability = float(self.state.num_hacks) / float(self.state.num_games)
        return hack_probability

    def is_attack_legal(self, attack_action:int) -> bool:
        """
        Check if a given attack is legal or not.

        :param attack_action: the attack to verify
        :return: True if legal otherwise False
        """
        import gym_idsgame.envs.util.idsgame_util as util
        return util.is_attack_id_legal(attack_action, self.ransomgame_config.game_config, self.state.attacker_pos,
                                       self.state, self.past_positions,
                                       past_reconnaissance_activities = self.past_reconnaissance_activities)

    def is_defense_legal(self, defense_action: int) -> bool:
        """
        Check if a given defense is legal or not.

        :param defense_action: the defense action to verify
        :return: True if legal otherwise False
        """
        import gym_idsgame.envs.util.idsgame_util as util
        return util.is_defense_id_legal(defense_action, self.ransomgame_config.game_config, self.state)

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

    def get_hack_reward(self, attack_type : int, node_id : int) -> Union[int, int]:
        """
        Returns the attacker and defender reward in the case when the hacker manages to reach the target node

        :return: (attacker_reward, defender_reward)
        """
        import gym_idsgame.envs.util.idsgame_util as util
        if not self.ransomgame_config.game_config.dense_rewards and not self.ransomgame_config.game_config.dense_rewards_v2 \
                and not self.ransomgame_config.game_config.dense_rewards_v3:
            return constants.GAME_CONFIG.POSITIVE_REWARD, -constants.GAME_CONFIG.POSITIVE_REWARD
        elif self.ransomgame_config.game_config.dense_rewards_v3:
            missed_defense = 0
            for defense in self.defenses:
                match = False
                for attack in self.attacks:
                    # if attack[0] == defense[0] and defense[2]:
                    #     match = True
                    if not defense[2] and attack[0] == defense[0] and defense[1] == attack[1]:
                        match = True
                if not match:
                    missed_defense += 1
            defense_score = util.defense_score(self.state, self.ransomgame_config.game_config.network_config,
                                               self.ransomgame_config.game_config)
            reference_defense_score = util.defense_score(self.ransomgame_config.game_config.initial_state,
                                                         self.ransomgame_config.game_config.network_config,
                                                         self.ransomgame_config.game_config)
            max_defense_score = self.ransomgame_config.game_config.max_value * self.ransomgame_config.game_config.network_config.num_rows

            attack_row, attack_col = self.state.attacker_pos
            row_ids = self.ransomgame_config.game_config.network_config.get_row_ids(attack_row)
            min_ats = self.state.min_attack_type(node_id, row_ids)
            if node_id in self.state.reconnaissance_actions and attack_type in min_ats:
                num_good_attacks = 1
                if (node_id, attack_type) in self.failed_attacks:
                    num_good_attacks = self.failed_attacks[(node_id, attack_type)] + 1
                num_bad_attacks = 0
                for k in self.failed_attacks:
                    n_id, at = k
                    if not n_id == node_id:
                        num_bad_attacks = num_bad_attacks + 1
                reward = max(num_good_attacks - math.pow(num_bad_attacks, 2), 0)

                return reward,-((max_defense_score-reference_defense_score)-(defense_score-reference_defense_score))

            return -constants.GAME_CONFIG.POSITIVE_REWARD, -((max_defense_score-reference_defense_score)-(defense_score-reference_defense_score))
        else:
            bonus = 1 if self.num_failed_attacks == 0 else 1 / self.num_failed_attacks
            return bonus + constants.GAME_CONFIG.POSITIVE_REWARD, -len(self.hacked_nodes)

    def get_detect_reward(self, target_node_id : int, attack_type : int, detection_value, reconnaissance : bool) -> Union[int, int]:
        """
        Returns the attacker and defender reward in the case when the attacker was detected.

        :return: (attacker_reward, defender_reward)
        """
        import gym_idsgame.envs.util.idsgame_util as util
        if not self.ransomgame_config.game_config.dense_rewards and not self.ransomgame_config.game_config.dense_rewards_v2 \
                and not self.ransomgame_config.game_config.dense_rewards_v3:
            return -constants.GAME_CONFIG.POSITIVE_REWARD, constants.GAME_CONFIG.POSITIVE_REWARD
        elif self.ransomgame_config.game_config.dense_rewards and self.ransomgame_config.game_config.dense_rewards_v2:
            return -100*constants.GAME_CONFIG.POSITIVE_REWARD, 100*constants.GAME_CONFIG.POSITIVE_REWARD
        elif self.ransomgame_config.game_config.dense_rewards_v3:
            missed_defense = 0
            for defense in self.defenses:
                match = False
                for attack in self.attacks:
                    # if attack[0] == defense[0] and defense[2]:
                    #     match = True
                    if not defense[2] and attack[0] == defense[0] and defense[1] == attack[1]:
                        match = True
                if not match:
                    missed_defense += 1

            attack_row, attack_col = self.state.attacker_pos
            row_ids = self.ransomgame_config.game_config.network_config.get_row_ids(attack_row)
            min_ats = self.state.min_attack_type(target_node_id, row_ids)

            added_defense = 0
            for defense in self.defenses:
                # if defense[2] and defense[0] == target_node_id:
                #     added_defense += 1
                if not defense[2] and defense[1] == attack_type and defense[0] == target_node_id:
                    added_defense += 1

            defense_score = util.defense_score(self.state, self.ransomgame_config.game_config.network_config, self.ransomgame_config.game_config)
            reference_defense_score = util.defense_score(self.ransomgame_config.game_config.initial_state,
                                                         self.ransomgame_config.game_config.network_config,
                                                         self.ransomgame_config.game_config)
            if not reconnaissance and target_node_id in self.state.reconnaissance_actions and attack_type in min_ats:
                return 0, (defense_score-reference_defense_score)
            else:
                return -constants.GAME_CONFIG.POSITIVE_REWARD, (defense_score-reference_defense_score)
        else:
            added_detection = 0
            for defense in self.defenses:
                if defense[2] and defense[0] == target_node_id:
                    added_detection += 1
            detection_ratio = added_detection/((detection_value+1))
            blocked_attacks = 0
            for defense in self.defenses:
                blocked = False
                for attack in self.attacks:
                    if defense[0] == attack[0] and defense[1] == attack[1] and defense[3] < attack[2]:
                        blocked = True
                if blocked:
                    blocked_attacks += 1
            norm_factor = self.state.game_step if self.state.game_step > 0 else 1
            defender_reward = (blocked_attacks)/norm_factor
            #return 0*constants.GAME_CONFIG.POSITIVE_REWARD, added_detection
            #return -1 * constants.GAME_CONFIG.POSITIVE_REWARD, added_detection
            #return -constants.GAME_CONFIG.POSITIVE_REWARD, defender_reward
            # min_at = self.state.min_attack_type(target_node_id)
            # if target_node_id in self.state.reconnaissance_actions and min_at == attack_type:
            #     #print("min_at3")
            #     return 0, constants.GAME_CONFIG.POSITIVE_REWARD
            return -constants.GAME_CONFIG.POSITIVE_REWARD, constants.GAME_CONFIG.POSITIVE_REWARD
            #else:
                #return -constants.GAME_CONFIG.POSITIVE_REWARD, constants.GAME_CONFIG.POSITIVE_REWARD
            #return -constants.GAME_CONFIG.POSITIVE_REWARD,(self.idsgame_config.game_config.num_nodes - len(self.hacked_nodes))

    def get_successful_attack_reward(self, attack_type : int, node_id : int) -> Union[int, int]:
        """
        Returns the reward for the attacker and defender after a successful attack on some server in
        the network

        :return:(attacker_reward, defender_reward)
        """
        if not self.ransomgame_config.game_config.dense_rewards and not self.ransomgame_config.game_config.dense_rewards_v2 \
                and not self.ransomgame_config.game_config.dense_rewards_v3:
            return 0, 0
        elif self.ransomgame_config.game_config.dense_rewards and self.ransomgame_config.game_config.dense_rewards_v2:
            attack_row, attack_col = self.state.attacker_pos
            if attack_row < self.furthest_hack:
                self.furthest_hack = attack_row
                return constants.GAME_CONFIG.POSITIVE_REWARD, -constants.GAME_CONFIG.POSITIVE_REWARD
            elif attack_row > self.furthest_hack:
                return -constants.GAME_CONFIG.POSITIVE_REWARD, constants.GAME_CONFIG.POSITIVE_REWARD
            return 0,0
        elif self.ransomgame_config.game_config.dense_rewards_v3:
            attack_row, attack_col = self.state.attacker_pos
            row_ids = self.ransomgame_config.game_config.network_config.get_row_ids(attack_row)
            min_ats = self.state.min_attack_type(node_id, row_ids)
            if node_id in self.state.reconnaissance_actions and attack_type in min_ats:
                num_good_attacks = 1
                if (node_id, attack_type) in self.failed_attacks:
                    num_good_attacks = self.failed_attacks[(node_id, attack_type)] + 1
                num_bad_attacks = 0
                for k in self.failed_attacks:
                    n_id, at = k
                    if not n_id == node_id:
                        num_bad_attacks = num_bad_attacks + 1
                reward = max(num_good_attacks - math.pow(num_bad_attacks, 2), 0)
                return reward, 0
            return -constants.GAME_CONFIG.POSITIVE_REWARD, 0
        else:
            attack_row, attack_col = self.state.attacker_pos
            bonus = 1 if self.num_failed_attacks == 0 else 1/self.num_failed_attacks
            if attack_row < self.furthest_hack:
                self.furthest_hack = attack_row
                extra_reward = 0
                if self.ransomgame_config.extra_reconnaissance_reward:
                    for rec_act in self.past_reconnaissance_activities:
                        node_id, rec_type = rec_act
                        server_id = self.ransomgame_config.game_config.network_config.get_node_id(self.state.attacker_pos)
                        if node_id == server_id:
                            extra_reward = 1
                return bonus + extra_reward + constants.GAME_CONFIG.POSITIVE_REWARD, 0
            elif attack_row > self.furthest_hack:
                return -constants.GAME_CONFIG.POSITIVE_REWARD, 0
            return 0, 0

    def validate_config(self, idsgame_config: IdsGameConfig) -> None:
        """
        Validates the configuration for the environment

        :param idsgame_config: the config to validate
        :return: None
        """
        if idsgame_config.game_config.num_layers < 0:
            raise AssertionError("The number of layers cannot be less than 0")
        if idsgame_config.game_config.num_attack_types < 1:
            raise AssertionError("The number of attack types cannot be less than 1")
        if idsgame_config.game_config.max_value < 1:
            raise AssertionError("The max attack/defense value cannot be less than 1")

    def get_blocked_attack_reward(self, target_node_id : int, attack_type : int) -> Union[int, int]:
        """
        Returns the reward for the attacker and defender after a blocked attack on some server in
        the network

        :return:(attacker_reward, defender_reward)
        """
        if not self.ransomgame_config.game_config.dense_rewards and not self.ransomgame_config.game_config.dense_rewards_v2:
            return 0, 0
        elif self.ransomgame_config.game_config.dense_rewards and not self.ransomgame_config.game_config.dense_rewards_v2:
            return 0, 0
        else:
            upd_defenses = []
            match = False
            for defense in self.defenses:
                if ((defense[0] == target_node_id and defense[1] == attack_type)) \
                        and not match:
                   match = True
                else:
                    upd_defenses.append(defense)
            self.defenses = upd_defenses
            if match:
                return 0, constants.GAME_CONFIG.POSITIVE_REWARD
            upd_defenses = []
            return 0, 0

    def get_observation(self) -> Union[np.ndarray, np.ndarray]:
        """
        Returns an observation of the state

        :return: (attacker_obs, defender_obs)
        """
        attacker_obs = self.state.get_attacker_observation(
            self.ransomgame_config.game_config.network_config, local_view=self.ransomgame_config.local_view_observations,
            reconnaissance=self.ransomgame_config.game_config.reconnaissance_actions,
        reconnaissance_bool_features=self.ransomgame_config.reconnaissance_bool_features)
        defender_obs = self.state.get_defender_observation(self.ransomgame_config.game_config.network_config)
        return attacker_obs, defender_obs

    def fully_observed(self) -> bool:
        """
        Boolean function to check whether the environment is configured to be fully observed or not

        :return: True if the environment is fully observed, otherwise false
        """
        return self.ransomgame_config.game_config.network_config.fully_observed

    def local_view_features(self) -> bool:
        """
        Boolean function to check whether the environment uses local view observations of the attacker

        :return: True if the environment uses local view observations
        """
        return self.ransomgame_config.local_view_observations

    def is_reconnaissance(self, action):
        import gym_idsgame.envs.util.idsgame_util as util
        server_id, server_pos, attack_type, reconnaissance = util.interpret_attack_action(action, self.ransomgame_config.game_config)
        return reconnaissance
        # if server_id not in self.state.reconnaissance_actions:
        #     #print("server_id:{}, rec actions:{}".format(server_id, self.state.reconnaissance_actions))
        #     return reconnaissance
        # else:
        #     #print("no rec needed")
        #     return False

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
        from gym_idsgame.envs.rendering.viewer import Viewer
        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, './rendering/', constants.RENDERING.RESOURCES_DIR)
        self.ransomgame_config.render_config.resources_dir = resource_path
        self.viewer = Viewer(idsgame_config=self.ransomgame_config)
        self.viewer.agent_start()

    def _build_state_to_idx_map(self):
        """
        Builds a map that maps states to index (useful when constructing Q-tables for example)

        :return: the lookup map
        """
        n_state_elems = self.ransomgame_config.game_config.num_nodes * \
                        (self.ransomgame_config.game_config.num_attack_types + 1)
        if self.ransomgame_config.game_config.network_config.fully_observed:
            n_state_elems = self.ransomgame_config.game_config.num_nodes * \
                            (self.ransomgame_config.game_config.num_attack_types + 1) * 2
        states = list(
            itertools.product(list(range(self.ransomgame_config.game_config.max_value + 1)), repeat=n_state_elems))
        assert int(len(states)) == int(math.pow(self.ransomgame_config.game_config.max_value + 1, n_state_elems))
        state_to_idx = {}
        for idx, s in enumerate(states):
            state_to_idx[s] = idx
        return state_to_idx


class AttackerEnv(RansomGameEnv, ABC):
    """
    Abstract AttackerEnv of the IdsGameEnv.

    Environments where the defender is part of the environment and the environment is designed to be used by an
    attacker-agent should inherit this class
    """

    def __init__(self, ransomgame_config: IdsGameConfig, save_dir: str = None, initial_state_path: str = None):
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
        import gym_idsgame.envs.util.idsgame_util as util
        attacker_action, _ = action
        return util.interpret_attack_action(attacker_action, self.ransomgame_config.game_config)

    def get_defender_action(self, action) -> Union[Union[int, int], int, int]:
        import gym_idsgame.envs.util.idsgame_util as util
        defend_id = self.ransomgame_config.defender_agent.action(self.state)
        defend_node_id, defend_node_pos, defend_type = util.interpret_defense_action(
            defend_id, self.ransomgame_config.game_config)
        return defend_node_id, defend_node_pos, defend_type

class IdsGameMinimalDefenseV0Env(AttackerEnv):
    """
    [AttackerEnv] 1 layer, 1 server per layer, 10 attack-defense-values, defender following the "defend minimal strategy"
    [Initial State] Defense: 2, Attack:0, Num vulnerabilities: 1, Det: 2, Vulnerability value: 0
    [Rewards] Sparse
    [Version] 0
    [Observations] partially observed
    [Environment] Deterministic
    [Attacker Starting Position] Start node
    [Reconnaissance activities] disabled
    [Reconnaissance bool features] No
    """
    def __init__(self, ransomgame_config: IdsGameConfig = None, save_dir: str = None, initial_state_path: str = None):
        """
        Initialization of the environment

        :param save_dir: directory to save outputs of the env
        :param initial_state_path: path to the initial state (if none, use default)
        :param ransomgame_config: configuration of the environment (if not specified a default config is used)
        """
        from gym_idsgame.agents.bot_agents.defend_minimal_value_bot_agent import DefendMinimalValueBotAgent
        if ransomgame_config is None:
            game_config = GameConfig(num_layers=1, num_servers_per_layer=1, num_attack_types=10, max_value=9)
            game_config.set_initial_state(defense_val=2, attack_val=0, num_vulnerabilities_per_node=1, det_val=2,
                                          vulnerability_val=0, num_vulnerabilities_per_layer=1)
            if initial_state_path is not None:
                game_config.set_load_initial_state(initial_state_path)
            defender_agent = DefendMinimalValueBotAgent(game_config)
            ransomgame_config = IdsGameConfig(game_config=game_config, defender_agent=defender_agent)
            ransomgame_config.render_config.caption = "idsgame-minimal_defense-v0"
        super().__init__(ransomgame_config=ransomgame_config, save_dir=save_dir)
import os
import time
from typing import Any, Union

import numpy as np
import tqdm

from gym_idsgame.agents.dao.experiment_result import ExperimentResult
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

        # self.env: RansomGameEnv = env
        self.state_to_idx = self.env.build_state_to_idx_map()

        self.Q_attacker = np.zeros((self.env.num_states_full, self.env.num_attack_actions))
        self.Q_defender = np.zeros((1, self.env.num_defense_actions))

        self.env.ransomgame_config.save_trajectories = False
        self.env.ransomgame_config.save_attack_stats = True

    # def get_state_id(self, observation: Any) -> int:
    #     """
    #     Convert a RansomGame attacker observation into a stable integer state id.
    #     """
    #     state_key = (
    #         # int(observation["time"]),
    #         tuple(int(x) for x in observation["stages"]),
    #     )
    #
    #     if state_key not in self.state_to_idx:
    #         next_state_id = len(self.state_to_idx)
    #
    #         if next_state_id >= self.Q_attacker.shape[0]:
    #             raise RuntimeError(
    #                 "RansomTabularQAgent discovered more states than Q_attacker was initialized for. "
    #                 "Increase env.num_states_full or switch Q_attacker to a dictionary-based table."
    #             )
    #
    #         self.state_to_idx[state_key] = next_state_id
    #
    #     return self.state_to_idx[state_key]

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

        # TODO change random sampling here
        if (np.random.rand() < self.config.epsilon and not eval) or (
            eval and np.random.random() < self.config.eval_epsilon
        ):
            return int(np.random.choice(legal_actions))

        best_action = max(legal_actions, key=lambda action: q_table[s][action])
        return int(best_action)

    def train(self) -> ExperimentResult:
        self.config.logger.info("Starting Training")
        self.config.logger.info(self.config.to_str())
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        done = False
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_train.set_description_str("[Train] epsilon: {:.2f}, avg_a_R: {:.2f}, avg_d_R: {:.2f}, "
                                             "avg_t: {:.2f}, avg_h: {:.2f}, acc_A_R: {:.2f}, "
                                             "acc_D_R: {:.2f}".format(self.config.epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Training
        for episode in range(self.config.num_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.render:
                    self.env.render(mode="human")

                if not self.config.attacker and not self.config.defender:
                    raise AssertionError("Must specify whether training an attacker agent or defender agent")

                # Default initialization
                s_idx_a = 0
                s_idx_d = 0
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    s_idx_a = self.env.get_state_id(attacker_obs)
                    if self.config.tab_full_state_space:
                        if self.env.fully_observed():
                            attacker_obs = np.append(attacker_obs, defender_obs)
                        t = tuple(attacker_obs.astype(int).flatten().tolist())
                        t = tuple(map(lambda x: min(x, self.max_value), t))
                        s_idx_a = self.state_to_idx[t]
                    attacker_action = self.get_action(s_idx_a, attacker=True)

                # if self.config.defender:
                #     s_idx_d = defender_state_node_id
                #     if self.config.tab_full_state_space:
                #         if self.env.fully_observed():
                #             defender_obs = np.append(attacker_obs, defender_obs)
                #         t = tuple(defender_obs.astype(int).flatten().tolist())
                #         t = tuple(map(lambda x: min(x, self.max_value), t))
                #         s_idx_d = self.state_to_idx[t]
                #     defender_action = self.get_action(s_idx_d, attacker=False)

                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _, info = self.env.step(action)
                reward, obs_prime, done = self.step_and_update(action, s_idx_a, s_idx_d, reward, obs_prime, done)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender

            # Render final frame
            if self.config.render:
                self.env.render(mode="human")

            # Record episode metrics
            self.num_train_games += 1
            self.num_train_games_total += 1
            if self.env.state.hacked:
                self.num_train_hacks += 1
                self.num_train_hacks_total += 1
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Log average metrics every <self.config.train_log_frequency> episodes
            if episode % self.config.train_log_frequency == 0:
                if self.num_train_games > 0 and self.num_train_games_total > 0:
                    self.train_hack_probability = self.num_train_hacks / self.num_train_games
                    self.train_cumulative_hack_probability = self.num_train_hacks_total / self.num_train_games_total
                else:
                    self.train_hack_probability = 0.0
                    self.train_cumulative_hack_probability = 0.0
                self.log_metrics(episode, self.train_result, episode_attacker_rewards, episode_defender_rewards,
                                 episode_steps, None, None, lr=self.config.alpha)
                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []
                self.num_train_games = 0
                self.num_train_hacks = 0

            # Run evaluation every <self.config.eval_frequency> episodes
            if episode % self.config.eval_frequency == 0:
                print("\n\n" + "-" * 50 + "\n\n")
                time.sleep(1.0)
                self.eval(episode)
                print("\n\n" + "-" * 50 + "\n\n")
                time.sleep(1.0)

            # Save Q table every <self.config.checkpoint_frequency> episodes
            if episode % self.config.checkpoint_freq == 0:
                self.save_q_table()
                self.env.save_trajectories(checkpoint = True)
                self.env.save_attack_data(checkpoint = True)
                if self.config.save_dir is not None:
                    time_str = str(time.time())
                    self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
                    self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

            # Reset environment for the next episode and update game stats
            done = False
            obs = self.env.reset(update_stats=True)
            attacker_obs, defender_obs = obs
            self.outer_train.update(1)

            # Anneal epsilon linearly
            self.anneal_epsilon()

        self.config.logger.info("Training Complete")

        # Final evaluation (for saving Gifs etc)
        print("\n\n" + "-" * 50 + "\n\n")
        time.sleep(1.0)
        self.eval(self.config.num_episodes, log=False)
        print("\n\n" + "-" * 50 + "\n\n")
        time.sleep(1.0)

        # Log and return
        self.log_state_values()

        # Save Q Table
        self.save_q_table()

        # Save other game data
        self.env.save_trajectories(checkpoint = False)
        self.env.save_attack_data(checkpoint = False)
        if self.config.save_dir is not None:
            time_str = str(time.time())
            self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

        return self.train_result

    def step_and_update(self, action, s_idx_a, s_idx_d, reward, obs_prime, done) -> Union[float, np.ndarray, bool]:
        attacker_reward, defender_reward = reward
        attacker_obs_prime, defender_obs_prime = obs_prime
        attacker_action, defender_action = action

        if self.config.attacker:
            s_prime_idx = 0
            if self.config.tab_full_state_space:
                if self.env.fully_observed():
                    attacker_obs_prime = np.append(attacker_obs_prime, defender_obs_prime)
                t = tuple(attacker_obs_prime.astype(int).flatten().tolist())
                t = tuple(map(lambda x: min(x, self.max_value), t))
                s_prime_idx = self.state_to_idx[t]
            self.q_learning_update(s_idx_a, attacker_action, attacker_reward, s_prime_idx, attacker=True)

        # if self.config.defender:
        #     s_prime_idx = 0
        #     if self.config.tab_full_state_space:
        #         if self.env.fully_observed():
        #             defender_obs_prime = np.append(attacker_obs_prime, defender_obs_prime)
        #         t = tuple(defender_obs_prime.astype(int).flatten().tolist())
        #         t = tuple(map(lambda x: min(x, self.max_value), t))
        #         s_prime_idx = self.state_to_idx[t]
        #     self.q_learning_update(s_idx_d, defender_action, defender_reward, s_prime_idx,
        #                            attacker=False)

        return reward, (attacker_obs_prime, defender_obs_prime), done


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

    def eval(self, train_episode, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param log: whether to log the result
        :param train_episode: train episode to keep track of logs and plots
        :return: None
        """
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

        self.num_eval_games = 0
        self.num_eval_hacks = 0

        self.eval_result = ExperimentResult()
        if self.config.eval_episodes < 1:
            return
        done = False

        # Video config
        # if self.config.video:
        #     if self.config.video_dir is None:
        #         raise AssertionError("Video is set to True but no video_dir is provided, please specify "
        #                              "the video_dir argument")
        #     self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
        #                               video_frequency=self.config.video_frequency)
        #     self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_eval = tqdm.tqdm(total=self.config.eval_episodes, desc='', position=1)
        # self.outer_eval.set_description_str(
        #     "[Eval] avg_a_R: {:.2f}, avg_d_R: {:.2f}, avg_t: {:.2f}, avg_h: {:.2f}, acc_A_R: {:.2f}, "
        #     "acc_D_R: {:.2f}".format(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Eval
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs

        # Get initial frame
        # if self.config.video or self.config.gifs:
        #     initial_frame = self.env.render(mode="rgb_array")[0]
        #     self.env.episode_frames.append(initial_frame)

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            attacker_state_values = []
            attacker_states = []
            attacker_frames = []
            defender_state_values = []
            defender_states = []
            defender_frames = []

            # if self.config.video or self.config.gifs:
            #     attacker_state_node_id = self.env.get_attacker_node_from_observation(attacker_obs)
            #     attacker_state_values.append(sum(self.Q_attacker[attacker_state_node_id]))
            #     attacker_states.append(attacker_state_node_id)
            #     attacker_frames.append(initial_frame)
            #     defender_state_node_id = 0
            #     defender_state_values.append(sum(self.Q_defender[defender_state_node_id]))
            #     defender_states.append(defender_state_node_id)
            #     defender_frames.append(initial_frame)

            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_state_node_id = 0
                defender_state_node_id = 0
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    s_idx_a = self.env.get_state_id(attacker_obs)
                    if self.config.tab_full_state_space:
                        if self.env.fully_observed():
                            attacker_obs = np.append(attacker_obs, defender_obs)
                        t = tuple(attacker_obs.astype(int).flatten().tolist())
                        t = tuple(map(lambda x: min(x, self.max_value), t))
                        s_idx_a = self.state_to_idx[t]
                    attacker_action = self.get_action(s_idx_a, attacker=True, eval=True)

                # if self.config.defender:
                #     s_idx_d = defender_state_node_id
                #     if self.config.tab_full_state_space:
                #         if self.env.fully_observed():
                #             defender_obs = np.append(attacker_obs, defender_obs)
                #         t = tuple(defender_obs.astype(int).flatten().tolist())
                #         t = tuple(map(lambda x: min(x, self.max_value), t))
                #         s_idx_d = self.state_to_idx[t]
                #     defender_action = self.get_action(s_idx_d, attacker=False, eval=True)

                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _, _ = self.env.step(action)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                attacker_obs_prime, defender_obs_prime = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = attacker_obs_prime
                defender_obs = defender_obs_prime

                # Save state values for analysis later
                # if self.config.video and len(self.env.episode_frames) > 1:
                #     if self.config.attacker:
                #         attacker_state_node_id = self.env.get_attacker_node_from_observation(attacker_obs)
                #         attacker_state_values.append(sum(self.Q_attacker[attacker_state_node_id]))
                #         attacker_states.append(attacker_state_node_id)
                #         attacker_frames.append(self.env.episode_frames[-1])
                #
                #     if self.config.defender:
                #         defender_state_node_id = 0
                #         defender_state_values.append(sum(self.Q_defender[defender_state_node_id]))
                #         defender_states.append(defender_state_node_id)
                #         defender_frames.append(self.env.episode_frames[-1])

            # Render final frame when game completed
            # if self.config.eval_render:
            #     self.env.render()
            #     time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {:>5}, Game ended after {} steps".format(episode, episode_step))

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Update eval stats
            self.num_eval_games +=1
            self.num_eval_games_total += 1
            self.eval_attacker_cumulative_reward += episode_attacker_reward
            self.eval_defender_cumulative_reward += episode_defender_reward
            if self.env.state.hacked:
                self.num_eval_hacks += 1
                self.num_eval_hacks_total += 1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_games > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                if self.num_eval_games_total > 0:
                    self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                        self.num_eval_games_total)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                                 episode_steps, update_stats=False, eval = True)

            # Save gifs
            # if self.config.gifs and self.config.video:
            #     self.env.generate_gif(self.config.gif_dir + "/episode_" + str(train_episode) + "_"
            #                           + time_str + ".gif", self.config.video_fps)

            if len(attacker_frames) > 1:
                # Save state values analysis for final state
                base_path = self.config.save_dir + "/state_values/" + str(train_episode) + "/"
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                np.save(base_path + "attacker_states.npy", attacker_states)
                np.save(base_path + "attacker_state_values.npy", attacker_state_values)
                np.save(base_path + "attacker_frames.npy", attacker_frames)


            if len(defender_frames) > 1:
                # Save state values analysis for final state
                base_path = self.config.save_dir + "/state_values/" + str(train_episode) + "/"
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                np.save(base_path + "defender_states.npy", np.array(defender_states))
                np.save(base_path + "defender_state_values.npy", np.array(defender_state_values))
                np.save(base_path + "defender_frames.npy", np.array(defender_frames))

            # Reset for new eval episode
            done = False
            obs = self.env.reset(update_stats=False)
            attacker_obs, defender_obs = obs
            # Get initial frame
            # if self.config.video or self.config.gifs:
            #     initial_frame = self.env.render(mode="rgb_array")[0]
            #     self.env.episode_frames.append(initial_frame)

            self.outer_eval.update(1)

        # Log average eval statistics
        if log:
            if self.num_eval_games > 0:
                self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
            if self.num_eval_games_total > 0:
                self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(self.num_eval_games_total)
            self.log_metrics(train_episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                             episode_steps, update_stats=True, eval=True)

        self.outer_eval.close()
        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result

    def log_state_values(self) -> None:
        """
        Utility function for printing the state-values according to the learned Q-function

        :return: None
        """
        if self.config.attacker:
            self.config.logger.info("--- Attacker State Values ---")
            for i in range(len(self.Q_attacker)):
                state_value = sum(self.Q_attacker[i])
                node_id = i
                self.config.logger.info("s: {}, V(s): {}".format(node_id, state_value))
            self.config.logger.info("--------------------")

        if self.config.defender:
            self.config.logger.info("--- Defender State Values ---")
            for i in range(len(self.Q_defender)):
                state_value = sum(self.Q_defender[i])
                node_id = i
                self.config.logger.info("s: {}, V(s): {}".format(node_id, state_value))
            self.config.logger.info("--------------------")

    def save_q_table(self) -> None:
        """
        Saves Q table to disk in binary npy format

        :return: None
        """
        time_str = str(time.time())
        if self.config.save_dir is not None:
            if self.config.attacker:
                path = self.config.save_dir + "/" + time_str + "_attacker_q_table.npy"
                self.config.logger.info("Saving Q-table to: {}".format(path))
                np.save(path, self.Q_attacker)
            if self.config.defender:
                path = self.config.save_dir + "/" + time_str + "_defender_q_table.npy"
                self.config.logger.info("Saving Q-table to: {}".format(path))
                np.save(path, self.Q_defender)
        else:
            self.config.logger.warning("Save path not defined, not saving Q table to disk")

"""
Configuration for the ransomgame environment
"""

from typing import Optional
from gym_idsgame.envs.dao.render_config import RenderConfig
from gym_ransomgame.envs.dao.game_config import GameConfig
from gym_idsgame.agents.agent import Agent


class RansomGameConfig:
    """
    DTO representing the configuration of the RansomGameEnv:
    """

    def __init__(
        self,
        render_config: RenderConfig,
        game_config: GameConfig,
        defender_agent: Optional[Agent] = None,
        attacker_agent: Optional[Agent] = None,
        initial_state_path: Optional[str] = None,
        save_trajectories: bool = False,
        save_attack_stats: bool = False,
        randomize_env: bool = False,
        ransomware_p: float = 0.5,
        local_view_observations: bool = False,
        # reconnaissance_actions : bool = False,
        # randomize_starting_position : bool = False,
        # reconnaissance_bool_features : bool = False,
        # extra_reconnaissance_reward : bool = False,
        # reconnaissance_reward : bool = False,
        randomize_visibility: bool = True,
        # reconnaissance_detection_factor = 1,
    ):
        """
        Constructor, initializes the config

        :param render_config: render config, e.g. colors, size, line width etc.
        :param game_config: game configuration, e.g. number of nodes
        :param defender_agent: the defender agent
        :param attacker_agent: the attacker agent
        :param initial_state_path: path to the initial state
        :param save_attack_stats: boolean flag whether to save attack statistics or not
        :param randomize_env: boolean flag whether to randomize the environment creation before each episode
        :param ransomware_p: probability that reset() makes the next episode a ransomware
                             episode rather than a benign one. 1.0 or 0.0 pins the episode
                             type and skips the draw entirely.
        :param local_view_observations: boolean flag whether features are provided in a "local view" mode
        :param randomize_visibility: whether to randomize visibilty during training (for partailly observed envs only)
        :param visibility_p: when randomizing visibility, set to visible with this probability
        """
        self.render_config = render_config
        self.game_config = game_config
        self.defender_agent = defender_agent
        self.attacker_agent = attacker_agent
        if self.render_config is None:
            self.render_config = RenderConfig()
        # if self.game_config is None:
        #     self.game_config = RansomGameConfig(initial_state_path=initial_state_path)
        # self.render_config.set_height(self.game_config.num_rows)
        # self.render_config.set_width(self.game_config.num_cols)
        self.save_trajectories = save_trajectories
        self.save_attack_stats = save_attack_stats
        self.randomize_env = randomize_env
        if not 0.0 <= ransomware_p <= 1.0:
            raise ValueError("ransomware_p must be in [0, 1], got {}".format(ransomware_p))
        self.ransomware_p = ransomware_p
        self.local_view_observations = local_view_observations
        # self.reconnaissance_actions = reconnaissance_actions
        # self.randomize_starting_position = randomize_starting_position
        # self.reconnaissance_bool_features = reconnaissance_bool_features
        # self.extra_reconnaissance_reward = extra_reconnaissance_reward
        # self.reconnaissance_reward = reconnaissance_reward
        self.randomize_visibility = randomize_visibility
        # self.visibility_p = visibility_p
        # self.reconnaissance_detection_factor = reconnaissance_detection_factor

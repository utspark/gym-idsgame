"""
A deterministic bot defense agent for the gym-ransomgame environment that raises the
alarm once the global lifecycle detector's score crosses a fixed level.
"""

from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_ransomgame.envs.dao.game_config import GameConfig
from gym_ransomgame.envs.dao.game_state import GameState


class ThresholdDefenderBotAgent(BotAgent):
    """
    Class implementing a deterministic defense policy that raises the alarm as soon as
    the global lifecycle detector's score reaches `global_detector_score_threshold`.

    The policy reads only a field that is part of the defender observation, so it is a
    valid opponent rather than one that peeks at hidden state. In particular it never
    sees the episode type: benign episodes leave the score at 0, so any threshold >= 1
    abstains on them without needing to be told.

    Reaching a full bar does not end the episode on its own, so the threshold has no
    upper-bound restriction the way an encryption-progress threshold would.
    """

    # The defender action space is Discrete(2) and the env treats the action as a
    # boolean alarm (see RansomGameEnv.step).
    NO_ALARM = 0
    ALARM = 1

    def __init__(self, game_config: GameConfig, global_detector_score_threshold: int = 3):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        :param global_detector_score_threshold: global detector score at or above which
            the alarm is raised
        """
        super(ThresholdDefenderBotAgent, self).__init__(game_config)
        if not 1 <= global_detector_score_threshold <= GameState.N_PROGRESS_STEPS:
            raise ValueError(
                "global_detector_score_threshold must be in [1, {}], got {}".format(
                    GameState.N_PROGRESS_STEPS, global_detector_score_threshold
                )
            )
        self.global_detector_score_threshold = global_detector_score_threshold

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        if game_state.global_detector_score >= self.global_detector_score_threshold:
            return self.ALARM
        return self.NO_ALARM

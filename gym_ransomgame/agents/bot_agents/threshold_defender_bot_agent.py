"""
A deterministic bot defense agent for the gym-ransomgame environment that raises the
alarm once the encryption progress bar crosses a fixed level.
"""

from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_ransomgame.envs.dao.game_config import GameConfig
from gym_ransomgame.envs.dao.game_state import GameState


class ThresholdDefenderBotAgent(BotAgent):
    """
    Class implementing a deterministic defense policy that raises the alarm as soon as
    the encryption progress bar reaches `threshold`.

    The policy reads only `encryption_level`, which is the whole of the defender
    observation, so it is a valid opponent rather than one that peeks at hidden state.
    In particular it never sees the episode type: benign episodes leave the bar at 0, so
    any threshold >= 1 abstains on them without needing to be told.

    The threshold must stay below GameState.N_PROGRESS_STEPS. A full bar means encryption
    already succeeded, which ends the episode as a hack before detection is evaluated, so
    a policy that waits for it can never fire.
    """

    # The defender action space is Discrete(2) and the env treats the action as a
    # boolean alarm (see RansomGameEnv.step).
    NO_ALARM = 0
    ALARM = 1

    def __init__(self, game_config: GameConfig, threshold: int = 2):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        :param threshold: encryption level at or above which the alarm is raised
        """
        super(ThresholdDefenderBotAgent, self).__init__(game_config)
        if not 1 <= threshold < GameState.N_PROGRESS_STEPS:
            raise ValueError(
                "threshold must be in [1, {}), got {}".format(
                    GameState.N_PROGRESS_STEPS, threshold
                )
            )
        self.threshold = threshold

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy.

        :param game_state: the game state
        :return: action_id
        """
        if game_state.encryption_level >= self.threshold:
            return self.ALARM
        return self.NO_ALARM

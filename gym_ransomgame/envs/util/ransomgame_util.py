"""
Utility functions for the gym-idsgame environment
"""
from typing import Any, TYPE_CHECKING, Union, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import io
import cv2
#from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig
#from gym_idsgame.envs.dao.game_config import GameConfig
#from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.node_type import NodeType
#from gym_idsgame.envs.dao.network_config import NetworkConfig

if TYPE_CHECKING:
    # ransomgame_env imports this module, so importing RansomGameEnv here at runtime
    # would form a cycle; only needed for the type hint below.
    from gym_ransomgame.envs.ransomgame_env import RansomGameEnv


def nonzero_q_table_states(
    env: "RansomGameEnv", q_table: np.ndarray, attacker: bool
) -> dict[int, dict[str, Any]]:
    """
    Maps each Q-table row with at least one nonzero entry to the observation it encodes.

    Rows that are all-zero were never visited during training, so only nonzero rows are
    worth decoding; visited-but-still-zero rows (e.g. a state only ever bootstrapped from)
    are indistinguishable from unvisited ones and are excluded along with them.

    :param env: the RansomGameEnv the table was trained against, for its state-id encoding
    :param q_table: Q_attacker or Q_defender, as saved by RansomTabularQAgent
    :param attacker: whether q_table is Q_attacker (True) or Q_defender (False)
    :return: mapping from nonzero row index to the observation fields that state id decodes to
    """
    nonzero_rows = np.flatnonzero(np.any(q_table != 0, axis=1))
    return {
        int(row): env.get_observation_from_state_id(int(row), attacker=attacker)
        for row in nonzero_rows
    }


def get_img_from_fig(fig, dpi=180):
    """
    Convert matplotlib fig to numpy array

    :param fig: fig to convert
    :param dpi: dpi of conversion
    :return: np array of the image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def action_dist_hist(data: np.ndarray,
                           title: str = "Test", xlabel: str = "test", ylabel: str = "test",
                           file_name: str = "test.eps", xlims: Union[float, float] = None) -> np.ndarray:
    """
    Plot a distribution of the policy

    :param data: the data to plot
    :param title: title of the plot
    :param xlabel: xlabel
    :param ylabel: ylabel
    :param file_name: path where to save file
    :param xlims: xlimits (optional)
    :return: numpy array of the figure
    """
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    if xlims is None:
        xlims = (min(data),
                 max(data))

    sns.distplot(data, kde=True,
              color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 0.5}, bins=xlims[1], fit=None)

    ax.set_xlim(xlims)
    ax.set_xticks(list(range(xlims[1]+1)))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set the grid on
    ax.grid('on')

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_size(10)
    ylab.set_size(10)

    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((.8, .8, .8))
    ax.spines['top'].set_color((.8, .8, .8))

    fig.tight_layout()
    fig.savefig(file_name + ".png", format="png", dpi=600)
    fig.savefig(file_name + ".pdf", format='pdf', dpi=600, bbox_inches='tight', transparent=True)
    data = get_img_from_fig(fig, dpi=100)
    plt.close(fig)
    return data


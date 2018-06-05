######################################################################
# Analytics utils
#
# Helper functions to analyze navigation performance
#
######################################################################


###################################
# Imports
#

from collections import namedtuple
import numpy as np
import pandas as pd
from utils.tree_navigation import is_sentence, is_ans_in_subtree, get_node_idx_path, \
    get_closest_idx_diff, find_closest_ans_node, navigation_dist_idx_path, get_evidence_length


###################################
# Globals
#

MetricsAgg = namedtuple('MetricsAgg', [
    'avg_acc',
    'avg_reward',
    'max_reward',
    'min_reward',
    'avg_q_diff',
    'avg_q_diff_win',
    'avg_path_len',
    'avg_illegal'
])


###################################
# Functions
#

def get_node_metrics(node, ans_line_idx):
    closest_ans_node, section_min_dist = find_closest_ans_node(node, ans_line_idx)
    if is_sentence(node):
        node_idx_path = get_node_idx_path(node.parent)
    else:
        node_idx_path = get_node_idx_path(node)
    ans_idx_path = get_node_idx_path(closest_ans_node)
    line_min_dist = navigation_dist_idx_path(node_idx_path, ans_idx_path)

    closest_line_idx, closest_line_diff = get_closest_idx_diff(node, [closest_ans_node.line])
    last_line = node.line - int(node.is_root)
    fsubtree = is_ans_in_subtree(node, ans_line_idx)

    node_metrics = [node_idx_path, ans_idx_path, section_min_dist, line_min_dist,
                    closest_line_diff, int(fsubtree), node.height, last_line]
    return node_metrics


def get_sample_metrics(node, ans_line_idx, evidence, t, path_num_illegal_moves, q_values, reward):
    last_node_metrics = get_node_metrics(node, ans_line_idx)
    evidence_metrics = [get_evidence_length(evidence), len(ans_line_idx), ans_line_idx[0]]
    navigation_metrics = [t, path_num_illegal_moves, max(q_values) - reward]

    return last_node_metrics + evidence_metrics + navigation_metrics


def get_metrics(metrics, reward_sums):
    df = pd.DataFrame(metrics, columns=('node_idx_path', 'ans_idx_path', 'section_min_dist', 'line_min_dist',
                                        'closest_line_diff', 'fsubtree', 'last_height', 'last_line',
                                        'evidence_len', 'num_line_idx', 'first_line_idx', 'num_steps', 'num_illegal', 'q_diff'))
    # accuracy stats
    acc_all = len(df[df.closest_line_diff == 0]) / len(df)

    # reward and q-value stats
    avg_reward = np.mean(reward_sums)
    max_reward = np.max(reward_sums)
    min_reward = np.min(reward_sums)
    avg_q_diff = df.q_diff.mean()
    avg_q_diff_win = df[df.closest_line_diff == 0].q_diff.mean()

    # navigation general stats
    df['illegal_frac'] = df.num_illegal / df.num_steps
    avg_illegal = df.illegal_frac.mean()
    avg_path_len = df.num_steps.mean()

    metrics_agg = MetricsAgg(
        avg_acc=acc_all,
        avg_reward=avg_reward,
        max_reward=max_reward,
        min_reward=min_reward,
        avg_q_diff=avg_q_diff,
        avg_q_diff_win=avg_q_diff_win,
        avg_path_len=avg_path_len,
        avg_illegal=avg_illegal)

    return metrics_agg, df


def metrics_agg_to_str(metrics_agg, fields=True):
    if fields:
        return "\t".join(["{} {:.4f}".format(key, getattr(metrics_agg, key)) for key in metrics_agg._fields[:-2]])
    else:
        return "\t".join(["{:.4f}".format(getattr(metrics_agg, key)) for key in metrics_agg._fields[:-2]])


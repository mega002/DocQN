######################################################################
# Tree Navigation
#
# Helper functions to navigate through evidence trees
#
######################################################################


###################################
# Imports
#

from utils.data_processing import PADDING_IDX
from anytree import Resolver, Walker, PreOrderIter
import numpy as np
import random

###################################
# Globals
#

ACTIONS = {'UPL':   0,
           'UPR':   1,
           'DOWN':  2,
           'LEFT':  3,
           'RIGHT': 4,
           'STOP':  5,
           'ANS':   6}

MAX_HEIGHT = 8
PROPS_DIM = (MAX_HEIGHT+1) * 2 + 5


###################################
# Helper Functions
#

def is_paragraph(node):
    return node.name[0] == 'p' and node.name[1].isdigit()


def is_sentence(node):
    return hasattr(node, 'sidx')


def is_leftmost(node):
    if node.is_root:
        return True

    children = node.parent.children
    node_idx = children.index(node)
    if node_idx == 0:
        return True

    return False


def is_rightmost(node):
    if node.is_root:
        return True

    children = node.parent.children
    node_idx = children.index(node)
    if node_idx == len(children)-1:
        return True

    return False


def get_tokens_txt(tokens):
    return [token[0] for token in tokens]


def to_add_prediction(node):
    return not node.is_root and np.random.rand() < 1.0 / len(ACTIONS)


###################################
# Functions
#

def get_closest_idx_diff(node, line_idx):
    diffs = np.array([abs(node.line - idx) + int(node.is_root) for idx in line_idx], dtype=np.int32)
    return line_idx[np.argmin(diffs)], np.min(diffs)


def find_closest_ans_node(node, ans_line_idx, section_level=True):
    unique_ans_line_idx = np.asarray(list(set(ans_line_idx)), dtype=np.int32)
    ans_nodes = [_locate_paragraph_node_by_line_idx(node.root, x) for x in unique_ans_line_idx]
    if section_level:
        navigation_dists = np.asarray([navigation_dist(node, x.parent) for x in ans_nodes], dtype=np.int32)
    else:
        navigation_dists = np.asarray([navigation_dist(node, x) for x in ans_nodes], dtype=np.int32)
    # taking the node at minimum distance with minimum line index
    # (unless its at 0 distance, then we take the node itself)
    min_dist = min(navigation_dists)
    if node.line in unique_ans_line_idx:
        closest_node = [x for x in ans_nodes if x.line == node.line][0]
    else:
        min_dist_line_idx = min(unique_ans_line_idx[navigation_dists == min_dist])
        min_idx = np.where(unique_ans_line_idx == min_dist_line_idx)[0][0]
        closest_node = ans_nodes[min_idx]

    return closest_node, min_dist


def get_node_line_span(node):
    # get start line
    span_s = node.line

    # get end line by walking to the bottom most right child
    tmp = node
    while tmp.height > 0:
        tmp = tmp.children[-1]
    span_e = tmp.line

    return span_s, span_e


def is_ans_in_subtree(node, line_idx):
    span_s, span_e = get_node_line_span(node)
    for idx in line_idx:
        if span_s <= idx <= span_e:
            return True
    return False


def get_evidence_title(sample):
    return sample['OrigEvidenceFilename'].rsplit('.', 1)[0]


def get_evidence_idx_title(sample, evidence_idx):
    return sample['EntityPages'][evidence_idx]['Filename'].rsplit('.', 1)[0]


def get_evidence_tree_height(evidence):
    return evidence['tree'].height


def get_evidence_length(evidence):
    # + 1 for evidence title
    return len(evidence["tokens"]) + 1


def get_evidence(evidence_dict, sample):
    evidence_title = get_evidence_title(sample)
    key = sample["QuestionId"] + '--' + evidence_title

    return evidence_dict.call(key)


def init_step(evidence, encoder, seq_len, observ_len, token_len, step):
    root = evidence["tree"]
    info = get_node_observ_props(root, step, evidence, encoder, seq_len, observ_len, token_len)
    (observ_w, observ_c, props, ans_w, ans_c, ans_p) = info

    return root, observ_w, observ_c, props


def init_step_random_balanced(evidence, encoder, predictions, seq_len, observ_len, token_len, max_episode_steps):
    if np.random.rand() < 0.2:
        nodes = [x for x in PreOrderIter(evidence['tree']) if not x.dummy]
    else:
        nodes = [x for x in PreOrderIter(evidence['tree']) if not x.dummy and not hasattr(x, 'sidx')]
    node = random.choice(nodes)

    if node.is_root:
        step = 0
    else:
        min_step = node.depth + get_node_dist(node.parent)[0]
        max_step = max_episode_steps-1  # minus 1 because counting from 0
        if min_step >= max_step:
            step = max_step
        else:
            step = np.random.randint(min_step, max_step)

    preds = predictions if to_add_prediction(node) else None
    info = get_node_observ_props(node, step, evidence, encoder, seq_len, observ_len, token_len, preds)
    (observ_w, observ_c, props, ans_w, ans_c, ans_p) = info

    return node, observ_w, observ_c, props, ans_w, ans_c, ans_p, step


def init_step_random_answer_radius(evidence, encoder, predictions, ans_line_idx, seq_len, observ_len, token_len, max_episode_steps,
                                   ans_radius, ans_dist_prob):
    if np.random.rand() > ans_dist_prob:
        return init_step_random_balanced(evidence, encoder, predictions, seq_len, observ_len, token_len, max_episode_steps)

    unique_ans_line_idx = list(set(ans_line_idx))
    random_ans_line = random.choice(unique_ans_line_idx)
    ans_node = _locate_paragraph_node_by_line_idx(evidence['tree'], random_ans_line)
    node = get_random_nearby_node(ans_node, max_actions=ans_radius, evidence=evidence)

    if node.is_root:
        step = 0
    else:
        min_step = node.depth + get_node_dist(node.parent)[0]
        max_step = max_episode_steps-1  # minus 1 because counting from 0
        if min_step >= max_step:
            step = max_step
        else:
            step = np.random.randint(min_step, max_step)

    preds = predictions if to_add_prediction(node) else None
    info = get_node_observ_props(node, step, evidence, encoder, seq_len, observ_len, token_len, preds)
    (observ_w, observ_c, props, ans_w, ans_c, ans_p) = info

    return node, observ_w, observ_c, props, ans_w, ans_c, ans_p, step


def get_random_nearby_node(start_node, max_actions, evidence):
    num_actions = np.random.randint(max_actions)

    node = start_node
    for _ in range(num_actions):
        legal_actions = get_legal_actions(node)
        legal_actions.remove(ACTIONS["STOP"])
        if ACTIONS["ANS"] in legal_actions:
            legal_actions.remove(ACTIONS["ANS"])
        if len(legal_actions) == 0:
            print("unexpected behavior - only STOP and or ANS are legal actions:\n\nstart_node: {}\t\nnode: {}".format(
                start_node, node))
            return node
        action = random.choice(legal_actions)
        node = _make_step(evidence, node, action)

    return node


def get_non_dummy_parent(node, evidence):
    if not node.dummy:
        return node

    else:
        if node.parent is None:
            print("Unexpected error, root is a dummy node: {}".format(evidence["title_tokens"]))
            return None

        elif not node.parent.dummy:
            return node.parent

        else:
            if node.parent.parent is None:
                print("Unexpected error, root is a dummy node: {}".format(evidence["title_tokens"]))
                return None

            elif not node.parent.parent.dummy:
                return node.parent.parent

            else:
                if node.parent.parent.parent is None:
                    print("Unexpected error, root is a dummy node: {}".format(evidence["title_tokens"]))
                    return None

                elif not node.parent.parent.parent.dummy:
                    return node.parent.parent.parent

                else:
                    print("Unexpected error, 4 dummy nodes: {}".format(evidence["title_tokens"]))
                    return None


def get_non_dummy_child(node, evidence):
    if not node.dummy:
        return node

    else:
        if node.children == ():
            print("Unexpected error, dummy node without children: {}".format(evidence["title_tokens"]))
            return None

        elif not node.children[0].dummy:
            return node.children[0]

        else:
            if node.children[0].children == ():
                print("Unexpected error, dummy node without children: {}".format(evidence["title_tokens"]))
                return None

            elif not node.children[0].children[0].dummy:
                return node.children[0].children[0]

            else:
                if node.children[0].children[0].children == ():
                    print("Unexpected error, dummy node without children: {}".format(evidence["title_tokens"]))
                    return None

                elif not node.children[0].children[0].children[0].dummy:
                    return node.children[0].children[0].children[0]

                else:
                    print("Unexpected error, 4 dummy nodes: {}".format(evidence["title_tokens"]))
                    return None


def make_step_upl(evidence, node):
    new_node = node

    if not node.is_root:
        new_node = make_step_left(evidence, node.parent)
        if new_node == node.parent:
            new_node = get_non_dummy_parent(new_node, evidence)

    return new_node


def make_step_upr(evidence, node):
    new_node = node

    if not node.is_root:
        new_node = make_step_right(evidence, node.parent)
        if new_node == node.parent:
            new_node = get_non_dummy_parent(new_node, evidence)

    return new_node


def make_step_down(evidence, node):
    new_node = node

    if node.children != ():
        new_node = get_non_dummy_child(node.children[0], evidence)
        if new_node is None:
            new_node = node

    return new_node


def make_step_left(evidence, node):
    new_node = node

    if not node.is_root:
        children = node.parent.children
        node_idx = children.index(node)
        if node_idx > 0:
            new_node = get_non_dummy_child(children[node_idx-1], evidence)
            if new_node is None:
                new_node = node

    return new_node


def make_step_right(evidence, node):
    new_node = node

    if not node.is_root:
        children = node.parent.children
        node_idx = children.index(node)
        if node_idx < len(children)-1:
            new_node = get_non_dummy_child(children[node_idx+1], evidence)
            if new_node is None:
                new_node = node

    return new_node


def make_step(evidence, encoder, predictions, node, action, seq_len, observ_len, token_len, step):
    if action == ACTIONS['UPL']:
        new_node = make_step_upl(evidence, node)

    elif action == ACTIONS['UPR']:
        new_node = make_step_upr(evidence, node)

    elif action == ACTIONS['DOWN']:
        new_node = make_step_down(evidence, node)

    elif action == ACTIONS['LEFT']:
        new_node = make_step_left(evidence, node)

    elif action == ACTIONS['RIGHT']:
        new_node = make_step_right(evidence, node)

    elif action == ACTIONS['ANS']:
        # can answer only from evidence content (not from the title)
        if not node.is_root:
            info = get_node_observ_props(node, step, evidence, encoder, seq_len, observ_len, token_len, predictions)
            (observ_w, observ_c, props, ans_w, ans_c, ans_p) = info
            done = False
            return node, observ_w, observ_c, props, ans_w, ans_c, ans_p, done
        new_node = node

    # STOP
    else:
        info = get_node_observ_props(node, step, evidence, encoder, seq_len, observ_len, token_len, predictions)
        (observ_w, observ_c, props, ans_w, ans_c, ans_p) = info
        done = True
        return node, observ_w, observ_c, props, ans_w, ans_c, ans_p, done

    info = get_node_observ_props(new_node, step, evidence, encoder, seq_len, observ_len, token_len)
    (observ_w, observ_c, props, ans_w, ans_c, ans_p) = info
    done = False

    return new_node, observ_w, observ_c, props, ans_w, ans_c, ans_p, done


def _make_step(evidence, node, action):
    if action == ACTIONS['UPL']:
        new_node = make_step_upl(evidence, node)

    elif action == ACTIONS['UPR']:
        new_node = make_step_upr(evidence, node)

    elif action == ACTIONS['DOWN']:
        new_node = make_step_down(evidence, node)

    elif action == ACTIONS['LEFT']:
        new_node = make_step_left(evidence, node)

    elif action == ACTIONS['RIGHT']:
        new_node = make_step_right(evidence, node)

    # STOP / ANS
    else:
        return node

    return new_node


def get_node_observ_props(node, step, evidence, encoder, seq_len, observ_len, token_len, predictions=None):
    observ_w, observ_c = get_node_observ(node, evidence, encoder, seq_len, observ_len, token_len)
    observ_w = np.reshape(observ_w, (1, observ_len))
    observ_c = np.reshape(observ_c, (1, observ_len, token_len))

    props = get_node_props(node, step)
    props = np.reshape(props, (1, len(props)))

    ans_w, ans_c, ans_p = None, None, None
    # getting predictions only for paragraph and sentence levels
    if predictions is not None and node.height <= 1:
        ans_w, ans_c, ans_p = get_node_prediction(node, evidence, encoder, predictions, token_len)
        ans_c = np.reshape(ans_c, (1, len(ans_c), token_len))
        ans_p = np.reshape(ans_p, (1, len(ans_p)))

    return observ_w, observ_c, props, ans_w, ans_c, ans_p


def get_node_observ(node, evidence, encoder, seq_len, observ_len, token_len):
    try:
        # add root observation
        if node.is_root:
            res_w, res_c = encoder.encode_seq(evidence['title_tokens'][:seq_len])
            res_c = encoder.pad_idx_seq_2dim(res_c, token_len, PADDING_IDX)

        # add paragraph observation
        elif is_paragraph(node):
            res_w_anc, res_c_anc = add_ancestors_observs(node, evidence, encoder, seq_len, token_len)
            res_w, res_c = encoder.encode_seq(evidence['tokens'][node.line][0][:seq_len])
            res_w = np.concatenate([res_w_anc, res_w])
            res_c = np.concatenate([res_c_anc, encoder.pad_idx_seq_2dim(res_c, token_len, PADDING_IDX)])

        # add sentence observation
        elif is_sentence(node):
            res_w_anc, res_c_anc = add_ancestors_observs(node.parent, evidence, encoder, seq_len, token_len)
            res_w, res_c = encoder.encode_seq(evidence['tokens'][node.line][node.sidx][:seq_len])
            res_w = np.concatenate([res_w_anc, res_w])
            res_c = np.concatenate([res_c_anc, encoder.pad_idx_seq_2dim(res_c, token_len, PADDING_IDX)])

        # add section observations
        else:
            # TODO: handle multi-sentence sections
            res_w_anc, res_c_anc = add_ancestors_observs(node, evidence, encoder, seq_len, token_len)
            res_w, res_c = encoder.encode_seq(evidence['tokens'][node.line][0][:seq_len])
            res_w = np.concatenate([res_w_anc, res_w])
            res_c = np.concatenate([res_c_anc, encoder.pad_idx_seq_2dim(res_c, token_len, PADDING_IDX)])

        observ_w = encoder.pad_idx_seq_1dim(res_w, observ_len, PADDING_IDX)
        observ_c = encoder.concate_pad_seq(res_c, observ_len, token_len, PADDING_IDX)

        return observ_w, observ_c

    except Exception as e:
        msg = "Error during 'get_node_observ': {}\nevidence: {}\nnode: {}\nnode_line: {}\nevidence_tokens_length: {}".format(
            e, evidence["tree"].name, node, node.line, len(evidence["tokens"]))
        print(msg)
        exit()


def add_ancestors_observs(node, evidence, encoder, seq_len, token_len):
    # assumes node is a title/headline node, namely not a paragraph nor a sentence
    observ_w, observ_c = [], []
    for anc_node in node.ancestors:
        if anc_node.dummy or is_paragraph(anc_node):
            continue
        if anc_node.is_root:
            res_w, res_c = encoder.encode_seq(evidence['title_tokens'][:seq_len])
            res_c = encoder.pad_idx_seq_2dim(res_c, token_len, PADDING_IDX)
        else:
            res_w, res_c = encoder.encode_seq(evidence['tokens'][anc_node.line][0][:seq_len])
            res_c = encoder.pad_idx_seq_2dim(res_c, token_len, PADDING_IDX)

        observ_w.extend(res_w)
        observ_c.extend(res_c)

    return observ_w, observ_c


def get_node_dist(node):
    dist_start, dist_end = 0, 0

    if not node.is_root:
        dist_start = node.parent.children.index(node)
        dist_end = len(node.parent.children) - 1 - node.parent.children.index(node)

    return dist_start, dist_end


def get_node_props(node, step):
    dist_start, dist_end = get_node_dist(node)
    if not node.is_root:
        dist_up_start, dist_up_end = get_node_dist(node.parent)
    else:
        dist_up_start, dist_up_end = 0, 0

    height = [0] * (MAX_HEIGHT+1)
    height[node.height] = 1
    depth = [0] * (MAX_HEIGHT+1)
    depth[node.depth] = 1

    return [step] + depth + height + [dist_start, dist_end, dist_up_start, dist_up_end]


def get_node_prediction(node, evidence, encoder, predictions, token_len):
    prediction = predictions[str(node.line)]
    ans_w, ans_c = encoder.encode_seq(prediction['tokens'])
    ans_c = encoder.pad_idx_seq_2dim(ans_c, token_len, PADDING_IDX)

    num_line_tokens = sum([len(sent) for sent in evidence['tokens'][node.line]])
    ans_p = [prediction['ent'], prediction['logits'], num_line_tokens]

    return ans_w, ans_c, ans_p


def get_node_prediction_text(node, predictions):
    if node.is_root or str(node.line) not in predictions:
        return None
    else:
        return predictions[str(node.line)]['texts']


def _locate_paragraph_node_by_line_idx(root, idx):
    nodes = [x for x in PreOrderIter(root) if not hasattr(x, 'sidx')]
    node_lines = [x.line if hasattr(x, 'line') else -1 for x in nodes]

    node_idx = len(node_lines) - 1 - node_lines[::-1].index(idx)
    ans_node_idx = node_idx if is_paragraph(nodes[node_idx]) else nodes.index(nodes[node_idx].parent)

    return nodes[ans_node_idx]


def get_node_section_idx(node):
    if node.is_root:
        return -1

    if node.parent.is_root:
        return node.parent.children.index(node)

    return node.anchestors[0].children.index(node.anchestors[1])


def is_illegal_move(node, action):
    if node.is_root:
        if action not in [ACTIONS['DOWN'], ACTIONS['STOP']]:
            return True
        return False

    if is_sentence(node) and action == ACTIONS['DOWN']:
        return True

    if is_leftmost(node) and action == ACTIONS['LEFT']:
        return True

    if is_rightmost(node) and action == ACTIONS['RIGHT']:
        return True

    return False


def get_legal_actions(node):
    return [ACTIONS[x] for x in ACTIONS if not is_illegal_move(node, ACTIONS[x])]


def navigation_dist(node1, node2):
    # navigation distance in steps from node1 to node2, including dummy nodes
    # it is valid to include them here because we look at the paths and not navigating
    # assuming the two node object are from the same tree instance
    path1 = np.asarray(get_node_idx_path(node1), dtype=np.int32)
    path2 = np.asarray(get_node_idx_path(node2), dtype=np.int32)
    return navigation_dist_idx_path(path1, path2)


def navigation_dist_idx_path(path1, path2):
    # navigation distance in steps from node at path1 to node at path2, including dummy nodes
    path1 = np.asarray(path1, dtype=np.int32)
    path2 = np.asarray(path2, dtype=np.int32)
    distance = 0
    n1 = len(path1)
    n2 = len(path2)

    if n1 <= n2:
        common_diff = np.abs(path1 - path2[:n1])
    else:
        common_diff = np.abs(path1[:n2] - path2)

    diff = np.argwhere(common_diff > 0)
    if len(diff) == 0:
        first_uncommon_idx = len(common_diff)
    else:
        first_diff_idx = diff[0][0]
        distance += common_diff[first_diff_idx]
        first_uncommon_idx = first_diff_idx + 1

    uncommon_path1 = path1[first_uncommon_idx:]
    uncommon_path2 = path2[first_uncommon_idx:]

    if len(uncommon_path1) > 0:
        distance += len(uncommon_path1)
    if len(uncommon_path2) > 0:
        distance += sum(uncommon_path2) + len(uncommon_path2)

    return distance


def get_node_idx_path(node):
    if node.is_root:
        return [0]

    return [get_node_dist(x)[0] for x in node.anchestors] + [get_node_dist(node)[0]]

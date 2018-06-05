######################################################################
# IO Utils
#
# Handles training configuration and logging
#
######################################################################


###################################
# Imports
#

from collections import namedtuple
import os
import argparse
import numpy as np
import pickle
import json
import random

from utils.tree_navigation import ACTIONS, get_node_dist, get_evidence_title

###################################
# Globals
#

LogConfig = namedtuple('LogConfig', [
    'log_trn_perf_navigator',
    'log_trn_stats_navigator',
    'dbg_log_perf_navigator',
    'navigator_output_path',
    'conf_path',
    'model_temp',
    'log_dir'
])

ModelConfig = namedtuple('ModelConfig', [
    'word_embedding_dim',
    'char_embedding_dim',
    'hidden_dim_q',
    'hidden_dim_x',
    'hidden_dim_a',
    'props_dim',
    'ans_props_dim',
    'output_dim',
    'dropout_rate',
    'learning_rate',
    'seq_length',
    'observ_length',
    'token_length'
])

TrainConfig = namedtuple('TrainConfig', [
    'batch_size',
    'max_steps',
    'replay_memory_init_size',
    'replay_memory_size',
    'per_alpha',
    'per_beta_start',
    'per_beta_end',
    'per_beta_growth_steps',
    'per_eps',
    'update_estimator_freq',
    'check_freq',
    'max_episode_steps',
    'epsilon_a_start',
    'epsilon_a_end',
    'epsilon_a_decay_steps',
    'epsilon_s_start',
    'epsilon_s_end',
    'epsilon_s_decay_steps',
    'gamma',
    'scores',
    'policy_type',
    'train_protocol',
    'combined_random_samples',
    'ans_radius',
    'ans_dist_prob'
])

RewardScores = namedtuple('RewardScores', [
    'r_delta',
    'r_win',
    'r_lose',
    'r_illegal'
])

CONF_PATH = '../logs/{}_rl.conf'
MODEL_TEMP = '../models/{}'
EVIDENCE_TEMP = '../data/evidence/wikipedia/{}'
VDEV_JSON = '../data/qa/verified-wikipedia-dev.json'
LOG_FILE_BUFF_SIZE = 1
TEST_SEED = 1618033988


###################################
# Functions
#

def valid_args(args):
    if args.resume or args.evaluate or args.test:
        if args.model_id is None or (args.model_step is None and not args.model_best):
            print("Both model id and step must be specified for resuming / evaluating a model")
            return False
        else:
            if args.model_best:
                model_step_path = os.path.join(MODEL_TEMP.format(args.model_id), args.model_id + '-best' + '.meta')
            else:
                model_step_path = os.path.join(MODEL_TEMP.format(args.model_id), args.model_id + '-' + str(args.model_step) + '.meta')

            if not os.path.isfile(model_step_path):
                print("There is no model with the given id: {}, step {}".format(args.model_id, args.model_step))
                return False

    return True


def get_configuration(word_embedding_dim, props_dim, args):
    if args.resume:
        model_conf, train_conf, log_conf, seed = load_execution_config(CONF_PATH.format(args.model_id))
        train_conf = TrainConfig(
            batch_size=train_conf.batch_size,
            max_steps=train_conf.max_steps + args.max_steps,
            replay_memory_init_size=train_conf.replay_memory_init_size,
            replay_memory_size=train_conf.replay_memory_size,
            per_alpha=train_conf.per_alpha,
            per_beta_start=train_conf.per_beta_start,
            per_beta_end=train_conf.per_beta_end,
            per_beta_growth_steps=train_conf.per_beta_growth_steps,
            per_eps=train_conf.per_eps,
            update_estimator_freq=train_conf.update_estimator_freq,
            check_freq=train_conf.check_freq,
            max_episode_steps=train_conf.max_episode_steps,
            epsilon_a_start=train_conf.epsilon_a_start,
            epsilon_a_end=train_conf.epsilon_a_end,
            epsilon_a_decay_steps=train_conf.epsilon_a_decay_steps,
            epsilon_s_start=train_conf.epsilon_s_start,
            epsilon_s_end=train_conf.epsilon_s_end,
            epsilon_s_decay_steps=train_conf.epsilon_s_decay_steps,
            gamma=train_conf.gamma,
            scores=train_conf.scores,
            policy_type=train_conf.policy_type,
            train_protocol=train_conf.train_protocol,
            combined_random_samples=train_conf.combined_random_samples,
            ans_radius=train_conf.ans_radius,
            ans_dist_prob=train_conf.ans_dist_prob
        )
        return log_conf, model_conf, train_conf, seed

    log_conf = LogConfig(
        log_trn_perf_navigator='../logs/{}_{}_trn_perf.log',    # model_id, init step (non zero when resuming training)
        log_trn_stats_navigator='../logs/{}_{}_trn_stats.log',  # model_id, init step (non zero when resuming training)
        dbg_log_perf_navigator='../logs/{}_{}_{}.dbg.log',         # model_id, evaluation step
        navigator_output_path='../logs/{}_{}_{}_output.json',   # model_id, init step (non zero when resuming training), name
        conf_path=CONF_PATH,
        model_temp=MODEL_TEMP,
        log_dir='../logs/'
    )

    model_conf = ModelConfig(
        word_embedding_dim=word_embedding_dim,
        char_embedding_dim=20,
        hidden_dim_q=300,
        hidden_dim_x=300,
        hidden_dim_a=300,
        props_dim=props_dim,
        ans_props_dim=3,
        output_dim=7,
        dropout_rate=1.0 if args.evaluate or args.test else args.keep_rate,
        learning_rate=args.learning_rate,
        seq_length=args.max_seq_len,
        observ_length=args.max_seq_len * 6,   # 6 levels
        token_length=args.max_token_len
    )

    rscores = RewardScores(
        r_delta=-0.02,
        r_win=1.0,
        r_lose=-1.0,
        r_illegal=-0.02
    )

    per_beta_growth_steps = args.per_beta_growth_steps if args.per_beta_growth_steps is not None else args.max_steps
    epsilon_a_decay_steps = args.epsilon_a_decay_steps if args.epsilon_a_decay_steps is not None else args.max_steps
    epsilon_s_decay_steps = args.epsilon_s_decay_steps if args.epsilon_s_decay_steps is not None else args.max_steps

    max_episode_steps = 100 if args.evaluate or args.test else 30
    train_conf = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        replay_memory_init_size=args.rm_init_size,
        replay_memory_size=args.rm_size,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_growth_steps=per_beta_growth_steps,
        per_eps=1e-6,
        update_estimator_freq=args.estimator_freq,
        check_freq=args.check_freq,
        max_episode_steps=max_episode_steps,
        epsilon_a_start=1.0,
        epsilon_a_end=0.1,
        epsilon_a_decay_steps=epsilon_a_decay_steps,
        epsilon_s_start=args.epsilon_s_start,
        epsilon_s_end=args.epsilon_s_end,
        epsilon_s_decay_steps=epsilon_s_decay_steps,
        gamma=0.996,
        scores=rscores,
        policy_type=args.policy_type,
        train_protocol=args.train_protocol,
        combined_random_samples=args.combined_random_samples,
        ans_radius=args.ans_radius,
        ans_dist_prob=args.ans_dist_prob
    )

    seed = args.seed if args.seed is not None else random.randrange(2 ** 32)
    if args.evaluate or args.test:
        seed = TEST_SEED

    return log_conf, model_conf, train_conf, seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False, help='Train a new model')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training of an existing model')
    parser.add_argument('--test', action='store_true', default=False, help='Test a trained model')
    parser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate a trained model')
    parser.add_argument('--model_id', default=None, help='Model id to resume training / evaluate', type=str)
    parser.add_argument('--model_step', default=None, help='Step of the model to resume training from / evaluate', type=int)
    parser.add_argument('--model_best', action='store_true', default=False, help='Evaluate the best trained model')
    parser.add_argument("--num_threads", type=int, default=-1, help="Number of CPU cores to use (maximum that is available if not set)")
    parser.add_argument("--seed", type=int, help="Random seed, default is random")
    parser.add_argument("--tfevents", action='store_true', default=False, help="Generate TF events for tensorboard")
    parser.add_argument("--rm_init_size", type=int, default=50000, help="Replay memory initial size, default is 50K")
    parser.add_argument("--rm_size", type=int, default=300000, help="Replay memory size, default is 300K")
    parser.add_argument("--per_beta_growth_steps", type=int, default=4000000,
                        help="Number of steps to increase beta, default is 4M")
    parser.add_argument("--epsilon_a_decay_steps", type=int, default=1000000,
                        help="Number of steps to decay epsilon_a, default is 1M")
    parser.add_argument("--epsilon_s_decay_steps", type=int, default=2000000,
                        help="Number of steps to decay epsilon_s, default is 2M")
    parser.add_argument("--epsilon_s_start", type=float, default=1.0, help="epsilon_s initial value, default is 1.0")
    parser.add_argument("--epsilon_s_end", type=float, default=0.5, help="epsilon_s final value, default is 0.5")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size, default is 64")
    parser.add_argument("--max_steps", type=int, default=4000000, help="Number of steps for training, default is 4M. \
                        When resuming training, it specifies how many steps will be added to the original number of steps.")
    parser.add_argument("--keep_rate", type=float, default=0.8, help="Keep rate for dropout, default is 0.8")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Initial learning rate, default is 0.0001")
    parser.add_argument("--max_seq_len", type=int, default=20, help="Maximum number of tokens in node observation")
    parser.add_argument("--max_token_len", type=int, default=50, help="Maximum number of characters in token")
    parser.add_argument("--estimator_freq", type=int, default=10000,
                        help="Frequency to update estimator parameters, default is 10K")
    parser.add_argument("--check_freq", type=int, default=20000,
                        help="Frequency to evaluate model performance, default is 20K")
    parser.add_argument('--policy_type', dest='policy_type', default='eglp', choices=['egp', 'eglp'], help='Default is eglp', type=str)
    parser.add_argument('--train_protocol', dest='train_protocol', default='sequential',
                        choices=['sequential', 'random_balanced', 'combined', 'combined_ans_radius'], help='Default is sequential', type=str)
    parser.add_argument("--combined_random_samples", type=int, default=5,
                        help="Number of random samples per iteration for combined training protocols, default is 5")
    parser.add_argument("--ans_radius", type=int, default=3,
                        help="Answer radius for training protocol combined_ans_radius, default is 3")
    parser.add_argument("--ans_dist_prob", type=float, default=0.5,
                        help="Random sampling probability for the training protocol combined_ans_radius, default is 0.5")
    return parser.parse_args()


def write_flog(text, flog):
    if flog is not None:
        flog.write(text)


def print_config(model, log_config):
    print("\nexecution config:\n----------------------------")
    print("model_id:\t{}\nseed:\t{}\n".format(model.model_id, model.seed))
    for config in [model.mc, model.tc, log_config]:
        for key in config._fields:
            print("{}:\t{}".format(key, getattr(config, key)))
        print("\n")
    print("----------------------------\n")


def store_execution_config(model, log_config):
    confs = [model.mc, model.tc, log_config]

    with open(log_config.conf_path.format(model.model_id) + '.txt', 'w') as fout:
        fout.write("model_id:\t{}\nseed:\t{}\n\n".format(model.model_id, model.seed))
        for config in confs:
            for key in config._fields:
                fout.write("{}:\t{}\n".format(key, getattr(config, key)))
            fout.write("\n")

    with open(log_config.conf_path.format(model.model_id), 'wb') as fout:
        pickle.dump(confs + [model.seed], fout)


def load_execution_config(conf_path):
    with open(conf_path, 'rb') as fconf:
        confs = pickle.load(fconf)
    return confs


def write_train_stats(step, reward_sums, path_avg_loss, path_avg_grads, path_lengths, flogstats):
    mean_rwrd, min_rwrd, max_rwrd = np.mean(reward_sums), np.min(reward_sums), np.max(reward_sums)
    avg_loss, avg_grd, avg_pl = np.mean(path_avg_loss), np.mean(path_avg_grads), np.mean(path_lengths)
    write_flog('{}\t{:.6f}\t{:.4e}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
        step, avg_loss, avg_grd, avg_pl, mean_rwrd, min_rwrd, max_rwrd), flogstats)
    print("step {}\tloss {:.6f}\tgrads {:.4e}\tavg_path_len {:.4f}\tavg.reward {:.4f}\tmin.reward {:.4f}\tmax.reward {:.4f}".format(
        step, avg_loss, avg_grd, avg_pl, mean_rwrd, min_rwrd, max_rwrd))


def write_predict_paths_header(flog):
    write_flog('evidence_title\tevidence_idx\tstep\tquestion_txt\tanswer\tobserv_line\tobserv_height\tobserv_depth\tobserv_level_idx\tobserv'
               + '\tobserv_wts\tquestion\tq_wts\tq_values\taction\treward\tans_line_idx\tdescription\n', flog)


def write_step_start_msg(flog, info, t, node, observ_tokens, x_weights, q_tokens, q_weights, q_values, action, reward):
    if flog is None:
        return

    (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = info
    q_values_str = ["{:.4f}".format(x) for x in q_values]
    x_weights_str = ["{:.4f}".format(x_weights[i]) for i in range(len(observ_tokens))]  # ignoring padding weights
    q_weights_str = ["{:.4f}".format(x) for x in q_weights.tolist()]
    evidence_title = "_".join([x[0] for x in evidence["title_tokens"]])
    msg = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
        evidence_title, eidx, t - 1, question_txt, answer_txt, node.line - node.is_root, node.height, node.depth, get_node_dist(node)[0],
        observ_tokens, x_weights_str, q_tokens, q_weights_str, q_values_str, action, reward, ans_line_idx)
    write_flog(msg, flog)


def write_step_end_msg(flog, node, next_node, action, prediction_text):
    if flog is None:
        return

    if (node == next_node and action != ACTIONS["ANS"]) or \
            (node.is_root and action == ACTIONS["ANS"]):
        write_flog("\tIllegal move\n", flog)

    elif not node.is_root and action == ACTIONS["ANS"]:
        assert prediction_text is not None
        write_flog("\tRaSoR output: {}\n".format(prediction_text), flog)

    else:
        write_flog("\t\n", flog)


def write_path_end_msg(flog, node, done, action, line_diff):
    if flog is None:
        return

    if done:
        if action == ACTIONS['STOP']:
            if node.is_root:
                write_flog("\tStopped at the title\n", flog)
            elif line_diff == 0:
                write_flog("\tWin!\n", flog)
            else:
                write_flog("\tStopped at wrong place\n", flog)
    else:
        write_flog("\tReached maximum steps\n", flog)


def create_json_record(sample, line_idx, from_raw=False, line_txt=None):
    if from_raw:
        context = line_txt
    else:
        context = sample['OrigEvidenceFilename'] + '_' + str(line_idx)

    record = {"title": get_evidence_title(sample) + '_' + str(line_idx),
              "paragraphs": [
                      {
                          "context": context,
                          "qas": [
                              {
                                  "id": sample['QuestionId'] + '--' + sample['OrigEvidenceFilename'] + '-->' + str(line_idx),
                                  "question": sample['Question']
                              }
                          ]
                      }
                  ]
              }

    return record


def get_evidence_line_idx_context(record):
    filename, line_idx = record["paragraphs"][0]["context"].rsplit('_', 1)
    line_idx = int(line_idx)

    if line_idx < 0:
        context = filename.rsplit('.', 1)[0].replace('_', ' ')

    else:
        evidence_filename = filename.replace('.txt', '.nop.txt')
        raw_evidence_path = EVIDENCE_TEMP.format(evidence_filename)

        with open(raw_evidence_path, 'r', encoding='utf-8') as fd:
            lines = fd.readlines()
            context = lines[line_idx].strip().strip('\n')

    return context


def write_predictions_json(records, output_path, from_raw=False):
    if output_path is None:
        return

    if not from_raw:
        for record in records:
            context = get_evidence_line_idx_context(record)
            record["paragraphs"][0]["context"] = context

    with open(output_path, 'w') as fd:
        json.dump({"data": records}, fd)


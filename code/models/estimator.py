######################################################################
# Estimator
#
# Main model
#
######################################################################


###################################
# Imports
#

import tensorflow as tf
from utils.analytics import *
from utils.io_utils import *
from utils.tree_navigation import *
from utils.data_processing import hash_token, PADDING
from models.network import *
from models.replay_buffer import State, StateExt, Transition, PrioritizedReplayBuffer
import random
from time import perf_counter


###################################
# Classes
#

class Model:
    def __init__(self, model_id, seed, log_config):
        self.model_id = model_id
        self.seed = seed
        self.init = None
        self.sess = None
        self.train_writer = None
        self.saver = None
        self.best_saver = None

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        model_dir_path = log_config.model_temp.format(self.model_id)
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        if not os.path.exists(log_config.log_dir):
            os.makedirs(log_config.log_dir)

    def start_sess(self, num_threads, tfevents=False):
        if num_threads != -1:
            config = tf.ConfigProto(intra_op_parallelism_threads=num_threads, inter_op_parallelism_threads=num_threads,
                                    allow_soft_placement=True, device_count={'CPU': 1})
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()
        if tfevents:
            self.train_writer = tf.summary.FileWriter('../logs')
            self.train_writer.add_graph(self.sess.graph)
        assert self.init is not None
        self.sess.run(self.init)

    def close_sess(self):
        if self.train_writer is not None:
            self.train_writer.close()
        self.sess.close()
        self.sess = None

    def load(self, step, log_config):
        assert self.sess is not None
        model_step_path = os.path.join(log_config.model_temp.format(self.model_id), self.model_id + '-' + str(step))
        self.saver.restore(self.sess, model_step_path)

    def store(self, step, log_config):
        assert self.sess is not None
        model_step_path = os.path.join(log_config.model_temp.format(self.model_id), self.model_id)
        save_path = self.saver.save(self.sess, model_step_path, global_step=step)
        return save_path

    def load_best(self, log_config):
        assert self.sess is not None
        model_step_path = os.path.join(log_config.model_temp.format(self.model_id), self.model_id + '-best')
        self.best_saver.restore(self.sess, model_step_path)

    def store_best(self, log_config):
        assert self.sess is not None
        model_step_path = os.path.join(log_config.model_temp.format(self.model_id), self.model_id + '-best')
        save_path = self.best_saver.save(self.sess, model_step_path)
        return save_path


class ModelEstimator(Model):
    def __init__(self, word_embeddings, char_emb_len, model_id, seed, model_conf, train_conf, log_config):
        Model.__init__(self, model_id, seed, log_config)
        self.mc = model_conf
        self.tc = train_conf
        self.lc = log_config
        self.step = 0
        self.epoch = 0
        self.best_acc = 0.0
        self.best_acc_dev = 0.0

        # estimators
        self.q_estimator = RLModel(known_emb=word_embeddings.known, unknown_emb=word_embeddings.unknown,
                                   char_emb_len=char_emb_len, model_conf=self.mc, scope='q_estimator')
        self.t_estimator = RLModel(known_emb=word_embeddings.known, unknown_emb=word_embeddings.unknown,
                                   char_emb_len=char_emb_len, model_conf=self.mc, scope='t_estimator')
        self.estimator_copy = ModelParametersCopier(self.q_estimator, self.t_estimator)

        # reply memory
        self.replay_memory = PrioritizedReplayBuffer(self.tc.replay_memory_size, alpha=self.tc.per_alpha)
        self.beta_schedule = np.linspace(self.tc.per_beta_start, self.tc.per_beta_end, self.tc.per_beta_growth_steps)

        # policy
        self.epsilon_a_schedule = np.linspace(self.tc.epsilon_a_start, self.tc.epsilon_a_end, self.tc.epsilon_a_decay_steps)
        if self.tc.policy_type == 'egp':
            self.policy = self.make_epsilon_greedy_policy(self.mc.output_dim)
        else:
            self.policy = self.make_epsilon_greedy_legal_policy(self.mc.output_dim)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.best_saver = tf.train.Saver(max_to_keep=1)

    def train(self, train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf):
        print('\npopulating replay memory...', end='')
        self.populate_replay_memory(train_samples, evidence_dict, encoder)
        print('done, {} transitions\n'.format(len(self.replay_memory)))
        print("model {} starts training".format(self.model_id))

        if self.tc.train_protocol.startswith("combined"):
            self.train_combined(train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf)

        elif self.tc.train_protocol == "random_balanced":
            self.train_random(train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf)

        else:
            self.train_sequential(train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf)

        print("model {} finished training".format(self.model_id))

    def train_combined(self, train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf):
        epsilon_s_schedule = np.linspace(self.tc.epsilon_s_start, self.tc.epsilon_s_end, self.tc.epsilon_s_decay_steps)

        while True:
            np.random.shuffle(train_samples)
            rewards, avg_grads, avg_loss = [], [], []
            reward_sums, path_avg_grads, path_avg_loss, path_lengths = [], [], [], []

            for sample in train_samples:
                # check if we're done
                if self.step >= self.tc.max_steps:
                    break

                sample_info = get_sample_info(sample, evidence_dict, encoder, self.mc.token_length)
                (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = sample_info
                epsilon_s = epsilon_s_schedule[min(self.step, self.tc.epsilon_s_decay_steps-1)]

                # random state sampling
                if np.random.rand() < epsilon_s:
                    for _ in range(self.tc.combined_random_samples):
                        protocol = self.tc.train_protocol if self.tc.train_protocol == "combined_ans_radius" else "random_balanced"
                        info = self.init_episode(protocol, evidence, encoder, predictions, None, ans_line_idx)
                        node, t = info[0], info[-1]
                        state = create_state(info, question_w, question_c)
                        next_node, next_state, done, reward, grad, loss = self.predict_sample_update(state, node, question_w, question_c,
                                                                                                     ans_line_idx, t, evidence, encoder, predictions)
                        if next_node is None and next_state is None:
                            continue

                        rewards.append(reward)
                        avg_grads.append(grad)
                        avg_loss.append(loss)

                        self.step += 1
                        self.post_update_checks(train_samples[:400], dev_samples[:400], evidence_dict, encoder,
                                                rewards, avg_loss, avg_grads, [-1], flogstats, flogperf)

                # sequential state sampling
                else:
                    t, done = 0, False
                    info = self.init_episode("sequential", evidence, encoder, predictions, t)
                    node = info[0]
                    state = create_state(info, question_w, question_c)
                    path_rewards, path_grads, path_loss = [], [], []
                    while True:
                        next_node, next_state, done, reward, grad, loss = self.predict_sample_update(state, node, question_w, question_c,
                                                                                                     ans_line_idx, t, evidence, encoder, predictions)
                        t += 1
                        if next_node is None and next_state is None:
                            break

                        path_rewards.append(reward)
                        path_grads.append(grad)
                        path_loss.append(loss)

                        node, state = next_node, next_state
                        self.step += 1
                        self.post_update_checks(train_samples[:400], dev_samples[:400], evidence_dict, encoder,
                                                reward_sums, path_avg_loss, path_avg_grads, path_lengths, flogstats, flogperf)

                        if done or t == self.tc.max_episode_steps:
                            break

                    reward_sums.append(np.sum(path_rewards))
                    path_avg_grads.append(np.mean(path_grads))
                    path_avg_loss.append(np.mean(path_loss))
                    path_lengths.append(t)

            save_path = self.store(self.step, self.lc)
            print("----- step {}\tmodel stored: {}".format(self.step, save_path))

            # check if we're done
            if self.step >= self.tc.max_steps:
                break

            print("----- step {}\tfinished epoch: {}".format(self.step, self.epoch + 1))
            self.epoch += 1

    def train_sequential(self, train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf):
        while True:
            np.random.shuffle(train_samples)
            reward_sums, path_avg_grads, path_avg_loss, path_lengths = [], [], [], []

            for sample in train_samples:
                # check if we're done
                if self.step >= self.tc.max_steps:
                    break

                sample_info = get_sample_info(sample, evidence_dict, encoder, self.mc.token_length)
                (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = sample_info
                t, done = 0, False
                info = self.init_episode("sequential", evidence, encoder, predictions, t)
                node = info[0]
                state = create_state(info, question_w, question_c)

                path_rewards, path_grads, path_loss = [], [], []
                while True:
                    next_node, next_state, done, reward, grad, loss = self.predict_sample_update(state, node, question_w, question_c,
                                                                                                 ans_line_idx, t, evidence, encoder, predictions)
                    t += 1
                    if next_node is None and next_state is None:
                        break

                    path_rewards.append(reward)
                    path_grads.append(grad)
                    path_loss.append(loss)

                    node, state = next_node, next_state
                    self.step += 1
                    self.post_update_checks(train_samples[:400], dev_samples[:400], evidence_dict, encoder,
                                            reward_sums, path_avg_loss, path_avg_grads, path_lengths, flogstats, flogperf)

                    if done or t == self.tc.max_episode_steps:
                        break

                reward_sums.append(np.sum(path_rewards))
                path_avg_grads.append(np.mean(path_grads))
                path_avg_loss.append(np.mean(path_loss))
                path_lengths.append(t)

            save_path = self.store(self.step, self.lc)
            print("----- step {}\tmodel stored: {}".format(self.step, save_path))

            # check if we're done
            if self.step >= self.tc.max_steps:
                break

            print("----- step {}\tfinished epoch: {}".format(self.step, self.epoch+1))
            self.epoch += 1

    def train_random(self, train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf):
        while True:
            np.random.shuffle(train_samples)
            rewards, avg_grads, avg_loss = [], [], []

            for sample in train_samples:
                # check if we're done
                if self.step >= self.tc.max_steps:
                    break

                sample_info = get_sample_info(sample, evidence_dict, encoder, self.mc.token_length)
                (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = sample_info

                # TODO: use init_step_batch instead? for more efficient sampling
                info = self.init_episode(self.tc.train_protocol, evidence, encoder, predictions, None, None)
                node, t = info[0], info[-1]
                state = create_state(info, question_w, question_c)
                next_node, next_state, done, reward, grad, loss = self.predict_sample_update(state, node, question_w, question_c,
                                                                                             ans_line_idx, t, evidence, encoder, predictions)
                if next_node is None and next_state is None:
                    continue

                rewards.append(reward)
                avg_grads.append(grad)
                avg_loss.append(loss)

                self.step += 1
                self.post_update_checks(train_samples[:400], dev_samples[:400], evidence_dict, encoder,
                                        rewards, avg_loss, avg_grads, [-1], flogstats, flogperf)

            save_path = self.store(self.step, self.lc)
            print("----- step {}\tmodel stored: {}".format(self.step, save_path))

            # check if we're done
            if self.step >= self.tc.max_steps:
                break

            print("----- step {}\tfinished epoch: {}".format(self.step, self.epoch + 1))
            self.epoch += 1

    def predict_paths(self, samples, evidence_dict, encoder, flog=None, fout=None):
        write_predict_paths_header(flog)
        metrics, reward_sums = [], []
        line_predictions = []

        for sample in samples:
            sample_info = get_sample_info(sample, evidence_dict, encoder, self.mc.token_length)
            (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = sample_info
            question_tokens = encoder.idxs_to_ws(question_w)

            t, done = 0, False
            info = self.init_episode("sequential", evidence, encoder, predictions, t)
            node, observ_w = info[0], info[1]
            state = create_state(info, question_w, question_c)
            observ_tokens = encoder.idxs_to_ws(observ_w)

            path_rewards, path_actions, path_num_illegal_moves = [], [], 0
            while True:
                action_probs, q_values, x_weights, q_weights = self.get_aprobs_qvals_weights(state, node, flog)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                path_actions.append(action)
                t += 1

                info = make_step(evidence, encoder, predictions, node, action, self.mc.seq_length,
                                 self.mc.observ_length, self.mc.token_length, t)
                next_node, observ_w, done = info[0], info[1], info[-1]
                next_state = create_state(info, question_w, question_c)
                reward = get_reward(node, action, ans_line_idx, evidence, self.tc.scores)
                path_rewards.append(reward)
                path_num_illegal_moves += int(is_illegal_move(node, action))

                write_step_start_msg(flog, sample_info, t, node, observ_tokens, x_weights, question_tokens, q_weights,
                                     q_values, action, reward)
                if done or t == self.tc.max_episode_steps:
                    break
                write_step_end_msg(flog, node, next_node, action, get_node_prediction_text(node, predictions))

                node, state = next_node, next_state
                observ_tokens = encoder.idxs_to_ws(observ_w)

            metrics.append(get_sample_metrics(node, ans_line_idx, evidence, t, path_num_illegal_moves, q_values, reward))
            closest_line_idx, closest_line_diff = get_closest_idx_diff(node, ans_line_idx)
            write_path_end_msg(flog, node, done, action, closest_line_diff)
            if fout is not None:
                line_predictions.append(create_json_record(sample, node.line - int(node.is_root)))

            reward_sums.append(sum(path_rewards))

        write_predictions_json(line_predictions, fout)

        return get_metrics(metrics, reward_sums)

    def predict_paths_test(self, samples, evidence_dict, encoder, fout):
        line_predictions = []
        total_time, total_steps = 0, 0

        for sample in samples:
            # start timer
            t0 = perf_counter()

            sample_info = get_sample_info(sample, evidence_dict, encoder, self.mc.token_length, test=True)
            (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = sample_info
            t, done = 0, False
            info = self.init_episode("sequential", evidence, encoder, predictions, t)
            node = info[0]
            state = create_state(info, question_w, question_c)

            while True:
                action_probs, q_values, x_weights, q_weights = self.get_aprobs_qvals_weights(state, node, None)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                t += 1

                info = make_step(evidence, encoder, predictions, node, action, self.mc.seq_length,
                                 self.mc.observ_length, self.mc.token_length, t)
                next_node, done = info[0], info[-1]
                next_state = create_state(info, question_w, question_c)

                if done or t == self.tc.max_episode_steps:
                    break

                node, state = next_node, next_state

            # stop timer
            total_time += perf_counter() - t0
            total_steps += t

            line_predictions.append(create_json_record(sample, node.line - int(node.is_root)))

        write_predictions_json(line_predictions, fout)

    def check_performance(self, train_samples, dev_samples, evidence_dict, encoder, flog):
        self.check_performance_samples(dev_samples, evidence_dict, encoder, 'DEV', flog)
        self.check_performance_samples(train_samples, evidence_dict, encoder, 'TRN', flog)

    def check_performance_samples(self, samples, evidence_dict, encoder, desc, flog):
        metrics_agg, df = self.predict_paths(samples, evidence_dict, encoder)
        msg = "{}\tstep {}\t".format(desc, self.step) + metrics_agg_to_str(metrics_agg)
        print('----- ' + msg)
        write_flog('{}\t{}\t'.format(desc, self.step) + metrics_agg_to_str(metrics_agg, fields=False) + '\n', flog)

        if metrics_agg.avg_acc > self.best_acc:
            self.best_acc = metrics_agg.avg_acc
            print("----- step {}\tnew best accuracy {:.4f} ({})".format(self.step, self.best_acc, desc))

        if desc == 'DEV' and metrics_agg.avg_acc > self.best_acc_dev:
            self.best_acc_dev = metrics_agg.avg_acc
            save_path = self.store_best(self.lc)
            print("----- step {}\tbest model stored: {}".format(self.step, save_path))

    def evaluate(self, samples, evidence_dict, encoder, perf_dbg_path, output_path):
        np.random.shuffle(samples)
        if perf_dbg_path is None:
            metrics_agg, df = self.predict_paths(samples, evidence_dict, encoder, None, output_path)
        else:
            with open(perf_dbg_path, 'w', encoding='utf-8') as flog:
                metrics_agg, df = self.predict_paths(samples, evidence_dict, encoder, flog, output_path)

        msg = metrics_agg_to_str(metrics_agg)
        print("model {} finished evaluation:\n{}".format(self.model_id, msg))

    def make_epsilon_greedy_policy(self, num_actions):
        estimator = self.q_estimator

        def policy_fn(sess, state, epsilon, get_weights=False):
            actions = np.ones(num_actions, dtype=float) * epsilon / num_actions
            if get_weights:
                q_values, x_weights, q_weights_context = estimator.predict(sess, state, get_weights)
            else:
                q_values = estimator.predict(sess, state, get_weights)

            q_values = q_values[0]
            best_action = np.argmax(q_values)
            actions[best_action] += (1.0 - epsilon)
            if get_weights:
                return actions, q_values, x_weights[0], q_weights_context[0]
            else:
                return actions, q_values

        return policy_fn

    def make_epsilon_greedy_legal_policy(self, num_actions):
        estimator = self.q_estimator

        def policy_fn(sess, state, node, epsilon, get_weights=False):
            legal_actions = get_legal_actions(node)
            num_legal_actions = len(legal_actions)
            actions = np.zeros(num_actions, dtype=float)
            actions[legal_actions] = epsilon / num_legal_actions
            if get_weights:
                q_values, x_weights, q_weights_context = estimator.predict(sess, state, get_weights)
            else:
                q_values = estimator.predict(sess, state, get_weights)

            q_values = q_values[0]
            best_action = np.argmax(q_values)
            actions[best_action] += (1.0 - epsilon)
            if get_weights:
                return actions, q_values, x_weights[0], q_weights_context[0]
            else:
                return actions, q_values

        return policy_fn

    def init_episode(self, protocol, evidence, encoder, predictions, t=None, ans_line_idx=None):
        if protocol == "sequential":
            return init_step(evidence, encoder, self.mc.seq_length, self.mc.observ_length,
                             self.mc.token_length, t)

        if protocol == "random_balanced":
            return init_step_random_balanced(evidence, encoder, predictions, self.mc.seq_length,
                                             self.mc.observ_length, self.mc.token_length, self.tc.max_episode_steps)

        if protocol == "combined_ans_radius":
            return init_step_random_answer_radius(evidence, encoder, predictions, ans_line_idx,
                                                  self.mc.seq_length, self.mc.observ_length, self.mc.token_length, self.tc.max_episode_steps,
                                                  self.tc.ans_radius, self.tc.ans_dist_prob)

    def predict_sample_update(self, state, node, question_w, question_c, ans_line_idx, t, evidence, encoder, predictions):
        action_probs, q_values = self.get_aprobs_qvals(state, node)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        t += 1

        try:
            info = make_step(evidence, encoder, predictions, node, action, self.mc.seq_length,
                             self.mc.observ_length, self.mc.token_length, t)
        except Exception as e:
            msg = "Error during 'make_step': {}\nevidence: {}\nnode: {}\naction: {}".format(
                e, evidence["tree"].name, node, action)
            print(msg)
            return None, None, None, None, None, None

        next_node, done = info[0], info[-1]
        next_state = create_state(info, question_w, question_c)
        reward = get_reward(node, action, ans_line_idx, evidence, self.tc.scores)
        self.replay_memory.add(Transition(state, action, reward, next_state,
                                          done or t == self.tc.max_episode_steps))

        # sample a batch from the replay memory, perform gradient descent update
        transitions = self.replay_memory.sample(self.tc.batch_size,
                                                beta=self.beta_schedule[min(self.step, self.tc.per_beta_growth_steps - 1)])
        (states_batch, action_batch, reward_batch, next_states_batch, done_batch, weights, idx_batch) = transitions

        # DDQN
        q_values_next = self.q_estimator.predict(self.sess, next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        q_values_next_target = self.t_estimator.predict(self.sess, next_states_batch)
        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.tc.gamma * \
                                       q_values_next_target[np.arange(self.tc.batch_size), best_actions]

        grad, loss, td_err = self.q_estimator.update(self.sess, states_batch, action_batch, targets_batch,
                                                     weights, self.train_writer, self.step)
        new_priorities = np.abs(td_err) + self.tc.per_eps
        self.replay_memory.update_priorities(idx_batch, new_priorities)

        return next_node, next_state, done, reward, grad, loss

    def post_update_checks(self, train_samples, dev_samples, evidence_dict, encoder,
                           rewards, avg_loss, avg_grads, avg_path_len, flogstats, flogperf):
        # target estimator periodic update
        if self.step % self.tc.update_estimator_freq == 0:
            self.estimator_copy.make(self.sess)
            print("----- step {}\testimator was updated".format(self.step))

        # writing training statistics
        if self.step % 2000 == 0 and len(rewards) > 0:
            write_train_stats(self.step, rewards, avg_loss, avg_grads, avg_path_len, flogstats)
            for lst in [rewards, avg_grads, avg_loss, avg_path_len]:
                lst.clear()

        # checking model performance
        if self.step % self.tc.check_freq == 0:
            self.check_performance(train_samples, dev_samples, evidence_dict, encoder, flogperf)

        # storing current model
        if self.step % 50000 == 0:
            save_path = self.store(self.step, self.lc)
            print("----- step {}\tmodel stored: {}".format(self.step, save_path))

    def get_aprobs_qvals(self, state, node):
        if self.tc.policy_type == 'egp':
            action_probs, q_values = self.policy(
                self.sess, state, self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
        else:
            action_probs, q_values = self.policy(
                self.sess, state, node, self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])

        return action_probs, q_values

    def get_aprobs_qvals_weights(self, state, node, flog):
        # no weights
        if flog is None:
            x_weights, q_weights = None, None
            action_probs, q_values = self.get_aprobs_qvals(state, node)

        # get weights
        else:
            if self.tc.policy_type == 'egp':
                action_probs, q_values, x_weights, q_weights = self.policy(
                    self.sess, state, self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)], get_weights=True)
            else:
                action_probs, q_values, x_weights, q_weights = self.policy(
                    self.sess, state, node, self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)], get_weights=True)

        return action_probs, q_values, x_weights, q_weights

    def populate_replay_memory(self, samples, evidence_dict, encoder):
        if self.tc.train_protocol.startswith("combined"):
            self.populate_replay_memory_combined(samples, evidence_dict, encoder)

        elif self.tc.train_protocol == "random_balanced":
            self.populate_replay_memory_random(samples, evidence_dict, encoder)

        else:
            self.populate_replay_memory_sequential(samples, evidence_dict, encoder)

    def populate_replay_memory_sequential(self, samples, evidence_dict, encoder):
        for sample in samples:
            sample_info = get_sample_info(sample, evidence_dict, encoder, self.mc.token_length)
            (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = sample_info

            t, done = 0, False
            info = self.init_episode("sequential", evidence, encoder, predictions, t)
            node = info[0]
            state = create_state(info, question_w, question_c)

            while len(self.replay_memory) < self.tc.replay_memory_init_size:
                if self.tc.policy_type == 'egp':
                    action_probs, q_values = self.policy(self.sess, state,
                                                         self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
                else:
                    action_probs, q_values = self.policy(self.sess, state, node,
                                                         self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                t += 1
                info = make_step(evidence, encoder, predictions, node, action, self.mc.seq_length,
                                 self.mc.observ_length, self.mc.token_length, t)
                next_node, done = info[0], info[-1]
                next_state = create_state(info, question_w, question_c)
                reward = get_reward(node, action, ans_line_idx, evidence, self.tc.scores)
                self.replay_memory.add(Transition(state, action, reward, next_state, done or t == self.tc.max_episode_steps))

                node, state = next_node, next_state
                if done or t == self.tc.max_episode_steps:
                    break

            if len(self.replay_memory) >= self.tc.replay_memory_init_size:
                break

    def populate_replay_memory_random(self, samples, evidence_dict, encoder):
        while len(self.replay_memory) < self.tc.replay_memory_init_size:
            # choose a random sample
            sample = random.choice(samples)
            sample_info = get_sample_info(sample, evidence_dict, encoder, self.mc.token_length)
            (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = sample_info

            # choose a random state
            info = self.init_episode(self.tc.train_protocol, evidence, encoder, predictions, None, None)
            node, t = info[0], info[-1]
            state = create_state(info, question_w, question_c)

            # choose an action
            # sample randomly if the model starts training, otherwise use pre-trained policy
            if self.step > 0:
                if self.tc.policy_type == 'egp':
                    action_probs, q_values = self.policy(self.sess, state,
                                                         self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
                else:
                    action_probs, q_values = self.policy(self.sess, state, node,
                                                         self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = np.random.choice(np.arange(self.mc.output_dim))
            t += 1

            # add a transition to memory
            info = make_step(evidence, encoder, predictions, node, action, self.mc.seq_length,
                             self.mc.observ_length, self.mc.token_length, t)
            next_node, done = info[0], info[-1]
            next_state = create_state(info, question_w, question_c)
            reward = get_reward(node, action, ans_line_idx, evidence, self.tc.scores)
            self.replay_memory.add(Transition(state, action, reward, next_state, done or t == self.tc.max_episode_steps))

    def populate_replay_memory_combined(self, samples, evidence_dict, encoder):
        epsilon_s_schedule = np.linspace(self.tc.epsilon_s_start, self.tc.epsilon_s_end, self.tc.epsilon_s_decay_steps)
        epsilon_s = epsilon_s_schedule[min(self.step, self.tc.epsilon_s_decay_steps - 1)]

        while len(self.replay_memory) < self.tc.replay_memory_init_size:
            # choose a random sample
            sample = random.choice(samples)
            sample_info = get_sample_info(sample, evidence_dict, encoder, self.mc.token_length)
            (question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions) = sample_info

            # random state sampling
            if np.random.rand() < epsilon_s:
                for _ in range(self.tc.combined_random_samples):
                    protocol = self.tc.train_protocol if self.tc.train_protocol == "combined_ans_radius" else "random_balanced"

                    info = self.init_episode(protocol, evidence, encoder, predictions, None, ans_line_idx)
                    node, t = info[0], info[-1]
                    state = create_state(info, question_w, question_c)

                    # choose an action
                    # sample randomly if the model starts training, otherwise use pre-trained policy
                    if self.step > 0:
                        if self.tc.policy_type == 'egp':
                            action_probs, q_values = self.policy(self.sess, state,
                                                                 self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
                        else:
                            action_probs, q_values = self.policy(self.sess, state, node,
                                                                 self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
                        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    else:
                        action = np.random.choice(np.arange(self.mc.output_dim))
                    t += 1

                    # add a transition to memory
                    info = make_step(evidence, encoder, predictions, node, action, self.mc.seq_length,
                                     self.mc.observ_length, self.mc.token_length, t)
                    next_node, done = info[0], info[-1]
                    next_state = create_state(info, question_w, question_c)
                    reward = get_reward(node, action, ans_line_idx, evidence, self.tc.scores)
                    self.replay_memory.add(Transition(state, action, reward, next_state, done or t == self.tc.max_episode_steps))

            # sequential state sampling
            else:
                t, done = 0, False
                info = self.init_episode("sequential", evidence, encoder, predictions, t)
                node = info[0]
                state = create_state(info, question_w, question_c)

                while len(self.replay_memory) < self.tc.replay_memory_init_size:
                    if self.tc.policy_type == 'egp':
                        action_probs, q_values = self.policy(self.sess, state,
                                                             self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
                    else:
                        action_probs, q_values = self.policy(self.sess, state, node,
                                                             self.epsilon_a_schedule[min(self.step, self.tc.epsilon_a_decay_steps - 1)])
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                    t += 1
                    info = make_step(evidence, encoder, predictions, node, action, self.mc.seq_length,
                                     self.mc.observ_length, self.mc.token_length, t)
                    next_node, done = info[0], info[-1]
                    next_state = create_state(info, question_w, question_c)
                    reward = get_reward(node, action, ans_line_idx, evidence, self.tc.scores)
                    self.replay_memory.add(Transition(state, action, reward, next_state, done or t == self.tc.max_episode_steps))

                    node, state = next_node, next_state
                    if done or t == self.tc.max_episode_steps:
                        break

                if len(self.replay_memory) >= self.tc.replay_memory_init_size:
                    break


class Encoder:
    def __init__(self, vocabulary):
        self.vocab = vocabulary

    def get_char_emb_len(self):
        return len(self.vocab.char_indices)

    def ws_to_idxs(self, words):
        seq_length = len(words)
        windices = np.zeros(seq_length, dtype=np.int32)
        for i in range(seq_length):
            windices[i] = self.w_to_idx(words[i][0])

        return windices

    def w_to_idx(self, word):
        if word in self.vocab.word_indices:
            return self.vocab.word_indices[word]
        else:
            return self.vocab.word_indices[hash_token(word)]

    def ws_to_c_idxs(self, words):
        seq_length = len(words)
        cindices = []
        for i in range(seq_length):
            cindices.append(self.w_to_c_idxs(words[i][0]))

        return cindices

    def w_to_c_idxs(self, word):
        return np.asarray([self.vocab.char_indices.get(word[j], 1) for j in range(len(word))], dtype=np.int32)

    def idxs_to_ws(self, indices):
        if indices.ndim > 1:
            tokens = []
            for indices_line in indices:
                tokens.append([self.vocab.index_words[index] for index in indices_line])
            tokens = tokens[0]
        else:
            tokens = [self.vocab.index_words[index] for index in indices]

        return [x for x in tokens if x != PADDING]

    def encode_seq(self, tokens):
        res_w = self.ws_to_idxs(tokens)
        res_c = self.ws_to_c_idxs(tokens)
        return res_w, res_c

    def pad_idx_seq_1dim(self, seq, seq_len, val):
        return np.pad(seq, (0, seq_len-len(seq)), 'constant', constant_values=(val,))

    def pad_idx_seq_2dim(self, seq, seq_len, val):
        return np.asarray([self.pad_idx_seq_1dim(x[:seq_len], seq_len, val) for x in seq], dtype=np.int32)

    def concate_pad_seq(self, seq, seq1_len, seq2_len, val):
        # assuming len(seq) <= seq1_len
        return np.concatenate([seq, np.ones((seq1_len - len(seq), seq2_len), dtype=np.int32) * val])


###################################
# Functions
#

def get_sample_info(sample, evidence_dict, encoder, token_len, test=False):
    question_w, question_c = encoder.encode_seq(sample['QuestionTokens'])
    question_c = encoder.pad_idx_seq_2dim(question_c, token_len, PADDING_IDX)
    question_c = np.reshape(question_c, (1, len(question_c), token_len))

    question_txt = sample['Question']
    eidx = sample['OrigEvidenceIdx']
    evidence, predictions = get_evidence(evidence_dict, sample)

    if test:
        answer_txt = None
        ans_line_idx = None
    else:
        answer_txt = sample['NormalizedAliases']
        ans_line_idx = sample['AnswerLineIdx']

    return question_w, question_c, question_txt, answer_txt, eidx, ans_line_idx, evidence, predictions


def create_state(info, question_w, question_c):
    if len(info) == 4:
        (node, observ_w, observ_c, props) = info
        state = State(q_w=question_w, q_c=question_c, x_w=observ_w, x_c=observ_c, p=props)
    else:
        (node, observ_w, observ_c, props, ans_w, ans_c, ans_p, step) = info
        if ans_w is None:
            state = State(q_w=question_w, q_c=question_c, x_w=observ_w, x_c=observ_c, p=props)
        else:
            state = StateExt(q_w=question_w, q_c=question_c, x_w=observ_w, x_c=observ_c, p=props,
                             a_w=ans_w, a_c=ans_c, a_p=ans_p)

    return state


def get_reward_line_diff(node, action, ans_line_idx, evidence, scores):
    navigation_reward = scores.r_delta

    if action == ACTIONS['STOP']:
        closest_idx, line_diff = get_closest_idx_diff(node, ans_line_idx)
        ev_len = get_evidence_length(evidence)
        navigation_reward = (ev_len - line_diff) / ev_len
        navigation_reward += scores.r_win if line_diff == 0 else 0

    if action == ACTIONS['ANS']:
        navigation_reward *= 3

    return navigation_reward


def get_reward(node, action, ans_line_idx, evidence, scores):
    return get_reward_line_diff(node, action, ans_line_idx, evidence, scores)



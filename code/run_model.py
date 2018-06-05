######################################################################
# Run Model
#
# Main routine for execution of model training and evaluation
#
######################################################################


###################################
# Imports
#

from models.estimator import *
from utils.rpc_client import EvidenceRpcClient
from utils.data_processing import WORD_EMBEDDING_DIM, DATA_CONFIG_NOP
from utils.analytics import MetricsAgg
from time import time
import pickle


###################################
# Globals
#

LOG_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, SEED = None, None, None, None
DATA_CONFIG_ = None


###################################
# Helper functions
#

def load_data():
    # load word embeddings
    print("Loading word embeddings...")
    with open(DATA_CONFIG_.glove_embeddings_path, 'rb') as fd:
        word_embeddings = pickle.load(fd)

    print("Loading vocabulary...")
    with open(DATA_CONFIG_.glove_vocab_path, 'rb') as fd:
        vocabulary = pickle.load(fd)
        encoder = Encoder(vocabulary)

    # get evidence RPC client
    print("Connect to evidence RPC server...")
    evidence_dict = EvidenceRpcClient()

    # load preprocessed datasets
    print("Loading datasets...")
    with open(DATA_CONFIG_.train_dataset.replace('.json', '.exp.pkl'), 'rb') as fd:
        train_samples = pickle.load(fd)
    with open(DATA_CONFIG_.dev_dataset.replace('.json', '.exp.pkl'), 'rb') as fd:
        dev_samples = pickle.load(fd)
    print("{} train samples, {} dev samples".format(len(train_samples), len(dev_samples)))

    return word_embeddings, encoder, evidence_dict, train_samples, dev_samples


def set_configuration(args):
    global LOG_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, SEED, DATA_CONFIG_
    LOG_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, SEED = get_configuration(WORD_EMBEDDING_DIM, PROPS_DIM, args)
    DATA_CONFIG_ = DATA_CONFIG_NOP


def create_model(word_embeddings, char_emb_len, model_id, train, step=0):
    model = ModelEstimator(word_embeddings, char_emb_len, model_id, SEED, MODEL_CONFIG, TRAIN_CONFIG, LOG_CONFIG)
    model.step = step
    log_trn_perf_navigator = LOG_CONFIG.log_trn_perf_navigator.format(model_id, step) if train else None
    log_trn_stats_path = LOG_CONFIG.log_trn_stats_navigator.format(model_id, step) if train else None

    return model, log_trn_perf_navigator, log_trn_stats_path


def evaluate_model(model, step, dev_samples, evidence_dict, encoder):
    log_dev_perf_dbg_path = LOG_CONFIG.dbg_log_perf_navigator.format(model.model_id, step, "dev")

    # zero exploration for evaluation
    model.step = model.tc.epsilon_a_decay_steps
    model.epsilon_a_schedule[-1] = 0.0

    print("Evaluating {} (step {}) on dev set...".format(model.model_id, step))
    output_path = LOG_CONFIG.navigator_output_path.format(model.model_id, step, "dev")
    model.evaluate(dev_samples, evidence_dict, encoder, log_dev_perf_dbg_path, output_path)


def test_model(model, step, evidence_dict, encoder):
    with open(DATA_CONFIG_.test_dataset.replace('.json', '.exp.pkl'), 'rb') as fd:
        samples = pickle.load(fd)

    # zero exploration for evaluation
    model.step = model.tc.epsilon_a_decay_steps
    model.epsilon_a_schedule[-1] = 0.0

    print("Test model {} (step {}) on test set...".format(model.model_id, step))
    output_path = LOG_CONFIG.navigator_output_path.format(model.model_id, step, "test")
    model.predict_paths_test(samples, evidence_dict, encoder, output_path)


###################################
# Main
#

def main():
    args = parse_args()
    if not valid_args(args):
        exit()
    set_configuration(args)
    word_embeddings, encoder, evidence_dict, train_samples, dev_samples = load_data()
    char_emb_len = encoder.get_char_emb_len()

    if args.train:
        print("\nCreating a model...")
        timestamp = str(time())
        model, log_trn_perf_path, log_trn_stats_path = create_model(word_embeddings, char_emb_len, timestamp, train=True)
        store_execution_config(model, LOG_CONFIG)
        print_config(model, LOG_CONFIG)
        print("Model ID: {}\ntotal params: {}".format(timestamp, model.q_estimator.get_num_model_params()))
        print("\nTraining...")
        model.start_sess(args.num_threads, args.tfevents)

        with open(log_trn_perf_path, 'w', LOG_FILE_BUFF_SIZE) as flogperf, \
                open(log_trn_stats_path, 'w', LOG_FILE_BUFF_SIZE) as flogstats:
            write_flog("dataset\tstep\t" + "\t".join(key for key in MetricsAgg._fields) + "\n", flogperf)
            write_flog("step\tloss\tgrads\tpath_len\tavg.reward\tmin.reward\tmax.reward\n", flogstats)
            model.train(train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf)
        print("\nFinished training.")

        evaluate_model(model, model.step, dev_samples, evidence_dict, encoder)
        model.close_sess()

    if args.resume:
        model, log_trn_perf_path, log_trn_stats_path = create_model(word_embeddings, char_emb_len, args.model_id, train=True, step=args.model_step)
        store_execution_config(model, LOG_CONFIG)
        print_config(model, LOG_CONFIG)
        print("Loading model to resume training: {} step {}".format(args.model_id, args.model_step))
        model.start_sess(args.num_threads)
        model.load(args.model_step, LOG_CONFIG)

        with open(log_trn_perf_path, 'w', LOG_FILE_BUFF_SIZE) as flogperf, \
                open(log_trn_stats_path, 'w', LOG_FILE_BUFF_SIZE) as flogstats:
            write_flog("dataset\tstep\t" + "\t".join(key for key in MetricsAgg._fields) + "\n", flogperf)
            write_flog("step\tloss\tgrads\tpath_len\tavg.reward\tmin.reward\tmax.reward\n", flogstats)
            model.train(train_samples, dev_samples, evidence_dict, encoder, flogstats, flogperf)
        print("\nFinished training.")

        evaluate_model(model, model.step, dev_samples, evidence_dict, encoder)
        model.close_sess()

    if args.evaluate:
        model, log_trn_perf_path, log_trn_stats_path = create_model(word_embeddings, char_emb_len, args.model_id, train=False)

        step = "best" if args.model_best else args.model_step
        print("Loading model for evaluation: {} step {}".format(args.model_id, step))
        model.start_sess(args.num_threads)
        if args.model_best:
            model.load_best(LOG_CONFIG)
        else:
            model.load(args.model_step, LOG_CONFIG)

        evaluate_model(model, step, dev_samples, evidence_dict, encoder)

        model.close_sess()

    if args.test:
        model, log_trn_perf_path, log_trn_stats_path = create_model(word_embeddings, char_emb_len, args.model_id, train=False)
        step = "best" if args.model_best else args.model_step
        print("Loading model for test: {} step {}".format(args.model_id, step))
        model.start_sess(args.num_threads)
        if args.model_best:
            model.load_best(LOG_CONFIG)
        else:
            model.load(args.model_step, LOG_CONFIG)

        test_model(model, step, evidence_dict, encoder)


if __name__ == '__main__':
    main()

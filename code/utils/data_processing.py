######################################################################
# Data processing
#
# Data structures and constants related to the preprocessed data
#
######################################################################


###################################
# Imports
#

from collections import namedtuple


###################################
# Globals
#

MAX_PROCESSES = 6  # maximum processes in pool at given time

WORD_VOCAB_SIZE = 100000
CHAR_THR = 49
NUM_UNK = 256
PADDING = "<PAD>"
PADDING_IDX = 0
UNK = ["<UNK{}>".format(i) for i in range(NUM_UNK)]

WORD_EMBEDDING_DIM = 300

MAX_ANS_OCC = 500

WordEmbeddings = namedtuple('WordEmbeddings', [
    'known',      # known words, including PADDING
    'unknown'     # unknown words
])

Vocabulary = namedtuple('Vocabulary', [
    'char_indices',
    'word_indices',
    'index_words'
])

DataConfig = namedtuple('DataConfig', [
    'glove_embeddings_path',
    'train_dataset',
    'dev_dataset',
    'test_dataset',
    'evidence_dir',
    'glove_vocab_path',
    'evidence_dict_path'
])

DATA_CONFIG_NOP = DataConfig(glove_embeddings_path="../data/glove.840B.300d.nop.pkl",
                             train_dataset="../data/qa/wikipedia-train.nop.json",
                             dev_dataset="../data/qa/wikipedia-dev.nop.json",
                             test_dataset="../data/qa/wikipedia-test-without-answers.nop.json",
                             evidence_dir="../data/evidence/wikipedia",
                             glove_vocab_path="../data/vocabulary_cased_glove.nop.pkl",
                             evidence_dict_path="../data/evidence_dict_cased.nop.pkl"
                             )


###################################
# Helper functions
#

def hash_token(token):
    return UNK[hash(token) % NUM_UNK]

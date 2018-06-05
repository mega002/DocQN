######################################################################
# Setup
#
# Download and extract TriviaQA-NoP raw and preprocessed data
#
######################################################################


###################################
# Imports
#

import os
import sys

###################################
# Globals
#

DATA_BASE_URL = 'https://www.cs.tau.ac.il/~taunlp/triviaqa-nop/{}'
DATA_BASE_DIR = '../data/{}'
FNAMES = ['triviaqa-nop.gz', 'triviaqa-nop-preprocessed.gz']
FDESCS = ['TriviaQA-NoP dataset', 'TriviaQA-NoP preprocessed data']


###################################
# Functions
#

def download_data():
    for i, fname in enumerate(FNAMES):
        if os.path.isfile(DATA_BASE_DIR.format(fname)):
            continue
        print('Downloading {}'.format(FDESCS[i]))
        wget_cmd = 'wget {} -O {}'.format(DATA_BASE_URL.format(fname), DATA_BASE_DIR.format(fname))
        if os.system(wget_cmd) != 0:
            print('Failure executing "{}"'.format(wget_cmd))
            sys.exit(1)


def extract_data():
    for i, fname in enumerate(FNAMES):
        print('Extracting {}'.format(FDESCS[i]))
        tar_cmd = 'tar -xzf {} -C {}'.format(DATA_BASE_DIR.format(FNAMES[i]), DATA_BASE_DIR.format(''))
        if os.system(tar_cmd) != 0:
            print('Failure executing "{}"'.format(tar_cmd))
            sys.exit(1)


def delete_gz_files():
    for i, fname in enumerate(FNAMES):
        if not os.path.isfile(DATA_BASE_DIR.format(fname)):
            continue
        print('Deleting {}'.format(DATA_BASE_DIR.format(fname)))
        wget_cmd = 'rm {}'.format(DATA_BASE_DIR.format(fname))
        if os.system(wget_cmd) != 0:
            print('Failure executing "{}"'.format(wget_cmd))
            continue


###################################
# Main
#

def main():
    if not os.path.exists(DATA_BASE_DIR.format('')):
        os.makedirs(DATA_BASE_DIR.format(''))
    download_data()
    extract_data()
    delete_gz_files()


if __name__ == '__main__':
    main()

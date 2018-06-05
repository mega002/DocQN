# Learning to Search in Long Documents Using Document Structure
This repo contains the code used in our paper [https://arxiv.org/abs/XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).
The code includes a framework for training and evaluation of the DocQN and DQN models on TriviaQA-NoP, our version of [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) where documents represented as tree objects.
The data is available for download [here](https://www.cs.tau.ac.il/~taunlp/triviaqa-nop/triviaqa-nop.gz).

There are two code version included, of the two baselines in the paper - full and coupled.
The full models leverage RaSoR predictions during navigation, while the coupled models do not.
All files ending with `_c.py` belong to the coupled version.


## Setup
The code requires python >= 3.5, tensorflow 1.3, and several other supporting libraries.
Tensorflow should be installed separately following the docs. To install the other dependencies use:
```bash
$ pip install -r requirements.txt`
```
Once the environment is set, you can download and extract the data by running the setup script:
```bash
$ python setup.py
```

Loading the data into memory requires at least 34GB RAM, where additional amount that depends on the replay memory size is required for training. To allow memory-efficient execution, which supports multiple executions in parallel, we run an RPC server that holds a single copy of the data in memory.
Running the RPC server is a requirement for the full models, and an option for the coupled models. To use it, [RabbitMQ](https://www.rabbitmq.com/install-debian.html) needs to be installed.

The code can run both on GPU and CPU devices.


## Data
TriviaQA-NoP comprises of dataset files and preprocessed files that are needed for code execution.
By running the setup script, as described above, all files will be downloaded and extracted into the `data` folder.

### TriviaQA-NoP dataset
The raw data is compressed in the `triviaqa-nop.gz` file, which comprises of raw evidence files without the preface section and their corresponding tree objects. In addition, the train, dev and test sets of TriviaQA (`json` files)
updated to the evidence files in TriviaQA-NoP.

### Preprocessed files
These include vocabulary and word embeddings based on [GloVe](https://nlp.stanford.edu/projects/glove/), per-paragraph [RaSoR](https://github.com/shimisalant/RaSoR) predictions, and "evidence dictionary" of question-evidence pairs that holds the data (tree objects, tokenized evidence files, etc.) to be loaded into memory during training and evaluation.
The `.exp.pkl` files under `data/qa` are an expanded version of the datasets (`json` files), where each sample of question and multiple evidences is broken into multiple question-evidence pairs.


## Running the RPC server
This step is a requirement for training and evaluation of DocQN/DQN models that use RaSoR predictions during navigation (e.g. the full models). For the coupled models, it is optional. To run the RPC server, execute the following command:
```bash
$ python run_rpc_server.py
```

It will start the server, which will keep running until being shut down (with Ctrl+C).


## Training

Use `run_model[_c].py` for training as follows:

```bash
$ PYTHONHASHSEED=[seed] python run_model[_c].py --train
```

Where [seed] is an integer that python's hash seed will be fixed to. We set up the PYTHONHASHSEED environment variable in this way, due to a usage of the python hash function in the code. Fixing PYTHONHASHSEED guarantees a consistent hash function across different executions and machines.

In order to use the RPC server in the coupled version, add the flag `--use_rpc`. There are plenty of configuration options that can be listed with the `--help` menu. One important argument is `--train_protocol` which controls the tree sampling method during training. Specifically, for training of DocQN, run:
```bash
$ PYTHONHASHSEED=[seed] python run_model[_c].py --train --train_protocol combined_ans_radius
```
and for training of DQN, run:
```bash
$ PYTHONHASHSEED=[seed] python run_model[_c].py --train --train_protocol sequential
```

During training, metrics of the navigation performance will be output, including navigation accuracy ('avg_acc').
Model checkpoints and logs will be stored under the `models` and `logs` folders, accordingly, where a unique id is generated for every model.

It is possible to resume training by using the `--resume` argument, together with `--model_id` and `--model_step`. Notice that the reply memory will be re-initialized in this case.


## Evaluation
Use `run_model[_c].py` for evaluation as follows:

```bash
$ PYTHONHASHSEED=[seed] python run_model[_c].py --evaluate --model_id [id] --model_best
```

For evaluation of a specific checkpoint, use `--model_step [step]` instead of `--model_best`.
This will evaluate the model on the development set of TriviaQA-NoP, and output two files:
* `logs/[model_id]_[model_step]_dev_output.json` - contains the selected paragraph for every question-evidence pair
in a [SQuAD format](https://rajpurkar.github.io/SQuAD-explorer/), that can be given as input to RaSoR (or any other reading comprehension model).
* `logs/[model_id]_[model_step]_dev_dbg.log` - a full navigation log, containing a description of the steps
performed by the model for every question-evidence pair

To obtain predictions for the test set, run:
```bash
$ PYTHONHASHSEED=[seed] python run_model[_c].py --test --model_id [id] --model_best
```
or:
```bash
$ PYTHONHASHSEED=[seed] python run_model[_c].py --test --model_id [id] --model_step [step]
```

Final answer predictions per question were obtained by running a version of [this implementation of RaSoR](https://github.com/shimisalant/RaSoR) on the model's output, and aggregating the predictions of multiple question evidences.  Currently, we are not publishing this version of RaSoR.

Please feel welcome to contact us for further details and resources.


## Pre-Trained Models
We release four pre-trained models:
* DocQN - 1524410969.1593015
* DQN - 1524411144.512193
* DocQN coupled - 1517547376.1149364
* DQN coupled - 1518010594.2258544

The models can be downloaded from [this](https://www.cs.tau.ac.il/~taunlp/triviaqa-nop/triviaqa-nop-pretrained-models.gz) link, and should be extracted to the `models` folder in the root directory.
Training and evaluation of these models were initiated with `PYTHONHASHSEED=1618033988`.
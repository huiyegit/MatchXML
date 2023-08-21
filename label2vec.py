# source code for MatchXML, 2022

import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
from gensim.models.callbacks import CallbackAny2Vec

import matplotlib.pyplot as plt
import copy
import spacy
import json
import numpy as np
import multiprocessing
from gensim.models import Word2Vec
import argparse

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

def parse_arguments():
    """Parse training arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_path",
        type=str,
        metavar="PATH",
        default="",
        help="path to the train labels(Y) file.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        metavar="PATH",
        default="",
        help="path to save the label embeddings.",
    )


    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        metavar="INT",
        help="Training algorithm: 1 for skip-gram; otherwise CBOW.",
    )

    parser.add_argument(
        "--sample",
        type=float,
        default=0.1,
        metavar="FLOAT",
        help="The threshold for configuring which higher-frequency words are randomly downsampled.",
    )

    parser.add_argument(
        "--ns_exponent",
        type=float,
        default=0.75,
        metavar="FLOAT",
        help=" The exponent used to shape the negative sampling distribution.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.025,
        metavar="FLOAT",
        help=" The initial learning rate.",
    )

    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.0001,
        metavar="FLOAT",
        help=" Learning rate will linearly drop to alpha-min.",
    )

    parser.add_argument(
        "--emb_size",
        type=int,
        default=100,
        metavar="INT",
        help="Dimensionality of the word vectors.",
    )

    parser.add_argument(
        "--negative",
        type=int,
        default=20,
        metavar="INT",
        help="the number of negative samples will be used.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="INT",
        help="the number of training epochs.",
    )

    parser.add_argument(
        "--win_size",
        type=int,
        default=20,
        metavar="INT",
        help="Maximum distance between target word and context word.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="INT",
        help="Seed for the random number generator.",
    )

    parser.add_argument(
        "--num_label",
        type=int,
        default=100,
        metavar="INT",
        help="Total number of labels.",
    )


    return parser


class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
            self.losses.append(loss)

        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
            self.losses.append(loss- self.loss_previous_step)
        self.epoch += 1
        self.loss_previous_step = loss

        if self.epoch % 10 == 0:
            self.plot(path="./w2v_training_loss.png")

    def plot(self, path):
        fig, (ax1) = plt.subplots(ncols=1, figsize=(6, 6))
        ax1.plot(self.losses, label="loss per epoch")
        plt.legend()
        plt.savefig(path)
        plt.close()
        print("Plotted loss.")


def label2vec_train(args):
    cores = multiprocessing.cpu_count()

    model_mode = args.mode
    sample_value = args.sample  # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    ns_exponent_value = args.ns_exponent  # [-1.0, -0.5, 0, 0.5, 1.0]
    lr = args.alpha  # [0.025, 0.0025, 0.00025]
    lr_min = args.alpha_min
    emb_size = args.emb_size  # [50, 100, 150]
    negative_value = args.negative  # [5, 10, 15, 20, 25]
    epoch_value = args.epochs  # [5, 10, 15, 20, 25, ..., 100, 150, 200]
    window_size = args.win_size

    seed_value = args.seed
    worker_num = cores - 4

    inputFileName = args.train_path
    lines = open(inputFileName, encoding="utf8")
    sentences = [row.strip().split(',') for row in lines]

    w2v_model = Word2Vec(min_count=1,
                         window=window_size,
                         size=emb_size,
                         sample=sample_value,
                         alpha=lr,
                         min_alpha=lr_min,
                         negative=negative_value,
                         sg=model_mode,
                         ns_exponent=ns_exponent_value,
                         workers=worker_num,
                         seed=seed_value,
                         compute_loss=True,
                         callbacks=[callback()])

    t = time()
    w2v_model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=epoch_value, report_delay=1,
                    compute_loss=True, callbacks=[callback()])
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.init_sims(replace=True)
    test = w2v_model.wv.most_similar(positive=["100"])
    print(test)

    return w2v_model


def get_weight_matrix(w2v_model, label_map, args):
    vocab_size = args.num_label
    dim = args.emb_size
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, dim), dtype=np.float32)
    for word, index in label_map.items():
        index = int(word)
        weight_matrix[index] = w2v_model.wv[word]
    return weight_matrix


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()

    w2v_model = label2vec_train(args)

    vocab = w2v_model.wv.vocab
    label_map = { }
    for i, k in enumerate(sorted(vocab.keys())):
        label_map[k] = i

    embedding_vectors = get_weight_matrix(w2v_model, label_map, args)
    file_path = args.save_path
    with open(file_path, 'wb') as f:
        np.save(f, embedding_vectors)





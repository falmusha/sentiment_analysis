from __future__ import absolute_import

import os
import io
import numpy as np
import pytreebank

from sklearn.model_selection import train_test_split


WORD2VEC = 'word2vec'
SKIPTHOUGHTS = 'skip-thoughts'

WORD2VEC_DIM = 300
SKIPTHOUGHTS_DIM = 4800

# ----------------------------------------- #
#
#                   UCI
#
# ----------------------------------------- #

UCI_DATASET_PATH ='./datasets/sentiment_labelled_sentences'
UCI_IMDB_FILEPATH = os.path.join(UCI_DATASET_PATH, 'imdb_labelled.txt')
UCI_YELP_FILEPATH = os.path.join(UCI_DATASET_PATH, 'yelp_labelled.txt')
UCI_AMAZON_FILEPATH = os.path.join(UCI_DATASET_PATH, 'amazon_cells_labelled.txt')
UCI_W2V_NPZ = './pds/uci_train_w2v.npz'
UCI_SKP_NPZ = './pds/uci_train_skp.npz'

# ----------------------------------------- #
#
#               STANFORD TREEBANK
#
# ----------------------------------------- #

STANFORD_TREEBANK_TRAIN = './datasets/stanford_sentiment_treebank/trees/train.txt'
STANFORD_TREEBANK_TRAIN_W2V_NPZ = './pds/stanford_treebank_train_w2v.npz'
STANFORD_TREEBANK_TRAIN_SKP_NPZ = './pds/stanford_treebank_train_skp.npz'
STANFORD_TREEBANK_VALID = './datasets/stanford_sentiment_treebank/trees/dev.txt'
STANFORD_TREEBANK_VALID_W2V_NPZ = './pds/stanford_treebank_valid_w2v.npz'
STANFORD_TREEBANK_VALID_SKP_NPZ = './pds/stanford_treebank_valid_skp.npz'
STANFORD_TREEBANK_TEST = './datasets/stanford_sentiment_treebank/trees/test.txt'
STANFORD_TREEBANK_TEST_W2V_NPZ = './pds/stanford_treebank_test_w2v.npz'
STANFORD_TREEBANK_TEST_SKP_NPZ = './pds/stanford_treebank_test_skp.npz'

# ----------------------------------------- #
#
#               STANFORD IMDB
#
# ----------------------------------------- #

STANFORD_IMDB_TRAIN_POS_DIR = './datasets/aclImdb/train/pos'
STANFORD_IMDB_TRAIN_NEG_DIR = './datasets/aclImdb/train/neg'
STANFORD_IMDB_TRAIN_W2V_NPZ = './pds/aclImdb_train_w2v.npz'
STANFORD_IMDB_TRAIN_SKP_NPZ = './pds/aclImdb_train_skp.npz'

STANFORD_IMDB_TEST_POS_DIR = './datasets/aclImdb/test/pos'
STANFORD_IMDB_TEST_NEG_DIR = './datasets/aclImdb/test/neg'
STANFORD_IMDB_TEST_W2V_NPZ = './pds/aclImdb_test_w2v.npz'
STANFORD_IMDB_TEST_SKP_NPZ = './pds/aclImdb_test_skp.npz'

nlp = None
skip_thoughts_model = None

def _init_skip_thoughts():
    global skip_thoughts_model

    if skip_thoughts_model:
        return

    global skipthoughts
    from skipthoughts import skipthoughts
    skip_thoughts_model = skipthoughts.load_model()

def _init_spacy():
    global nlp

    if nlp:
        return

    import spacy
    nlp = spacy.load('en')

def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
        user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def hot_encode_2_classes(labels):
    hot_l = np.zeros((labels.shape[0], 2))

    for i in range(labels.shape[0]):
        if labels[i] == 1:   # positive
            hot_l[i][0] = 1
            hot_l[i][1] = 0
        else:               # negative
            hot_l[i][0] = 0
            hot_l[i][1] = 1

    return hot_l


def merge_datasets(x1, y1, x2, y2):
    x1_len = x1.shape[0]
    x2_len = x2.shape[0]
    y1_len = y1.shape[0]
    y2_len = y2.shape[0]

    x = np.zeros((x1_len+x2_len, x1.shape[1]))
    y = np.zeros((y1_len+y2_len,))

    x[:x1_len] = x1
    x[x1_len:] = x2

    y[:y1_len] = y1
    y[y1_len:] = y2

    return x, y


def skip_thoughts_vecs(sentences):
    _init_skip_thoughts()

    print('---------------- skip_thoughts -------------------')
    return skipthoughts.encode(skip_thoughts_model, sentences, batch_size=512)


def word_2_vecs(sentences):
    _init_spacy()

    print('---------------- word2vec -------------------')
    vecs = np.zeros((len(sentences), WORD2VEC_DIM))
    for idx, s in enumerate(sentences):
        vecs[idx] = nlp(s).vector

    return vecs


def load_stanford_treebank_dataset():
    if os.path.isfile(STANFORD_TREEBANK_NPZ):
        data = np.load(STANFORD_TREEBANK_NPZ)
        print('loaded stanford treebank dataset from %s' % STANFORD_TREEBANK_NPZ)
        return data['x'], data['y']

    train = load_stanford_treebank_dataset_file(STANFORD_TREEBANK_TRAIN,
                                                STANFORD_TREEBANK_TRAIN_NPZ)
    valid = load_stanford_treebank_dataset_file(STANFORD_TREEBANK_VALID,
                                                STANFORD_TREEBANK_VALID_NPZ)
    test = load_stanford_treebank_dataset_file(STANFORD_TREEBANK_TEST,
                                                STANFORD_TREEBANK_TEST_NPZ)

    return train, valid, test


def load_stanford_treebank_dataset_file(path, npz_path):
    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        print('loaded stanford treebank dataset from %s' % npz_path)
        return (data['x'], data['y'])

    _init_spacy()
    dataset = pytreebank.import_tree_corpus(path)

    labels = list()
    word_vecs = list()

    for tree in dataset:
        for label, sentence in tree.to_labeled_lines():
            word_vecs.append(nlp(sentence).vector)
            if label >= 2:
                labels.append(1) # positive
            else:
                labels.append(0) # negative
    n = len(word_vecs)
    x = np.zeros((n, 300))
    y = np.zeros(n)

    for i in range(n):
        x[i] = word_vecs[i]
        y[i] = labels[i]

    np.savez(npz_path, x=x, y=y)

    return x, y


def load_stanford_imdb_dataset_file(path):
    sentences = list()
    labels = list()
    i = 0
    for f in os.listdir(path):
        if f.endswith('.txt'):
            filepath = os.path.join(path, f)
            with open(filepath) as f:
                sentences.append(f.read().decode('utf-8'))
            _, label = os.path.basename(filepath).replace('.txt', '').split('_')
            if int(label) >= 5:
                labels.append(1) # positive
            else:
                labels.append(0) # negative
            i += 1
            print('++++ FILE %s -> %s' % (path, os.path.basename(filepath)))

    print('++++ STANFORD IMDB DONE %s' % path)
    return sentences, np.array(labels)


def load_stanford_imdb_dataset(vec_rep=WORD2VEC):
    if vec_rep == WORD2VEC:
        train_npz_path = STANFORD_IMDB_TRAIN_W2V_NPZ
        test_npz_path = STANFORD_IMDB_TEST_W2V_NPZ
    else:
        train_npz_path = STANFORD_IMDB_TRAIN_SKP_NPZ
        test_npz_path = STANFORD_IMDB_TEST_SKP_NPZ

    if os.path.isfile(train_npz_path) and os.path.isfile(test_npz_path):
        train_data = np.load(train_npz_path)
        test_data = np.load(test_npz_path)
        print('loaded train uci dataset from %s' % train_npz_path)
        print('loaded test uci dataset from %s' % test_npz_path)
        return (train_data['x'], train_data['y']), \
               (test_data['x'], test_data['y'])

    print('---------------- STANFORD_IMDB_TRAIN_POS_DIR -------------------')
    x_pos_train, y_pos_train = \
            load_stanford_imdb_dataset_file(STANFORD_IMDB_TRAIN_POS_DIR)
    print('---------------- STANFORD_IMDB_TRAIN_NEG_DIR -------------------')
    x_neg_train, y_neg_train = \
            load_stanford_imdb_dataset_file(STANFORD_IMDB_TRAIN_NEG_DIR)
    print('---------------- STANFORD_IMDB_TEST_POS_DIR -------------------')
    x_pos_test, y_pos_test = \
            load_stanford_imdb_dataset_file(STANFORD_IMDB_TEST_POS_DIR)
    print('---------------- STANFORD_IMDB_TEST_NEG_DIR -------------------')
    x_neg_test, y_neg_test = \
            load_stanford_imdb_dataset_file(STANFORD_IMDB_TEST_NEG_DIR)

    if vec_rep == WORD2VEC:
        x_pos_train = word_2_vecs(x_pos_train)
        x_neg_train = word_2_vecs(x_neg_train)
        x_pos_test = word_2_vecs(x_pos_test)
        x_neg_test = word_2_vecs(x_neg_test)
    else:
        x_pos_train = skip_thoughts_vecs(x_pos_train)
        x_neg_train = skip_thoughts_vecs(x_neg_train)
        x_pos_test = skip_thoughts_vecs(x_pos_test)
        x_neg_test = skip_thoughts_vecs(x_neg_test)

    x_train, y_train = merge_datasets(x_pos_train,
                                      y_pos_train,
                                      x_neg_train,
                                      y_neg_train)

    x_test, y_test = merge_datasets(x_pos_test,
                                      y_pos_test,
                                      x_neg_test,
                                      y_neg_test)

    np.savez(train_npz_path, x=x_train, y=y_train)
    np.savez(test_npz_path, x=x_test, y=y_test)
    print('saving standford imdb train dataset to %s' % train_npz_path)
    print('saving standford imdb test dataset to %s' % train_npz_path)

    return (x_train, y_train)


def load_uci_dataset_from_file(filepath):
    inputs = list()
    labels = list()
    with open(filepath) as f:
        for line in f:
            line_list = line.split()
            label = int(line_list.pop(-1))
            sentence = ' '.join(line_list).decode('utf-8')

            inputs.append(sentence)
            labels.append(label)

    return (inputs, labels)


def load_uci_dataset(vec_rep=WORD2VEC):
    if vec_rep == WORD2VEC:
        npz_path = UCI_W2V_NPZ
    else:
        npz_path = UCI_SKP_NPZ

    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        print('loaded uci dataset from %s' % npz_path)
        return data['x'], data['y']

    imdb = load_uci_dataset_from_file(UCI_IMDB_FILEPATH)
    yelp = load_uci_dataset_from_file(UCI_YELP_FILEPATH)
    amzn = load_uci_dataset_from_file(UCI_AMAZON_FILEPATH)

    sentences = imdb[0] + yelp[0] + amzn[0]
    y = np.array(imdb[1] + yelp[1] + amzn[1])

    if vec_rep == WORD2VEC:
        x = word_2_vecs(sentences)
    else:
        x = skip_thoughts_vecs(sentences)

    np.savez(npz_path, x=x, y=y)
    print('saved uci dataset to %s' % npz_path)

    return x, y

if __name__ == '__main__':
    load_stanford_imdb_dataset(SKIPTHOUGHTS)

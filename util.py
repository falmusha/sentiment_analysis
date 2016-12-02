from __future__ import absolute_import

import os
import io
import numpy as np
import pytreebank

from multiprocessing import Pool
from sklearn.model_selection import train_test_split


WORD2VEC = 'word2vec'
SKIPTHOUGHTS = 'skip-thoughts'

WORD2VEC_DIM = 300
SKIPTHOUGHTS_DIM = 4800

# ----------------------------------------- #
#
#                   PROJ
#
# ----------------------------------------- #

W2V_SMALL_NPZ = './pds/small_w2v.npz'
W2V_ALL_NPZ = './pds/all_w2v.npz'

SKP_SMALL_NPZ = './pds/small_skp.npz'
SKP_ALL_NPZ = './pds/all_skp.npz'

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


def to_skip_thoughts_vec(data):
    vecs = skipthoughts.encode(skip_thoughts_model, data[0], batch_size=512)

    return (vecs, data[1])

def skip_thoughts_vecs(sentences):
    _init_skip_thoughts()

    print('---------------- skip_thoughts -------------------')

    vecs = skipthoughts.encode(skip_thoughts_model, sentences,
            batch_size=len(sentences))
    return vecs

    batch_size = 512
    data = list() # batches
    n = len(sentences)
    for i in xrange(0, n, batch_size):
        print('i = %d ---> START = %d, END = %d' %(i, i, i+batch_size))
        data.append((sentences[i:i+batch_size], labels[i:i+batch_size]))

    pool = Pool(20)
    try:
        pdata = pool.map(to_skip_thoughts_vec, data)
        pool.close()
        pool.join()
    except (KeyboardInterrupt, SystemExit):
        pool.close()
        pool.terminate()

    import pdb; pdb.set_trace()

    vecs = np.zeros((n, SKIPTHOUGHTS_DIM))
    _labels = np.zeros((n))
    for i, batch in enumerate(pdata):
        start = i * batch_size
        end = start + batch_size
        print('i = %d ---> START = %d, END = %d' %(i, start, end))
        vecs[start:end] = batch[0]
        _labels[start:end] = batch[1]

    return vecs, _labels


def to_nlp_word_vec(data):
    return (nlp(data[0]).vector, data[1])

def word_2_vecs(sentences, labels):
    _init_spacy()

    print('---------------- word2vec -------------------')

    data = list()
    n = len(sentences)
    for i in range(n):
        data.append((sentences[i], labels[i]))

    pool = Pool(20)

    try:
        pdata = pool.map(to_nlp_word_vec, data)
        pool.close()
        pool.join()
    except (KeyboardInterrupt, SystemExit):
        pool.close()
        pool.terminate()

    vecs = np.zeros((n, WORD2VEC_DIM))
    labels = np.zeros((n))
    for i in range(n):
        vecs[i] = pdata[i][0]
        labels[i] = pdata[i][1]

    return vecs, labels


def load_stanford_treebank_dataset(vec_rep=WORD2VEC):
    train_npz_path = STANFORD_TREEBANK_TRAIN_W2V_NPZ
    valid_npz_path = STANFORD_TREEBANK_VALID_W2V_NPZ
    test_npz_path = STANFORD_TREEBANK_TEST_W2V_NPZ

    if os.path.isfile(train_npz_path) \
        and os.path.isfile(valid_npz_path) \
        and os.path.isfile(test_npz_path):

        train_data = np.load(train_npz_path)
        valid_data = np.load(valid_npz_path)
        test_data = np.load(test_npz_path)
        print('loaded train stanford treebank dataset from %s' % train_npz_path)
        print('loaded valid stanford treebank dataset from %s' % valid_npz_path)
        print('loaded test stanford treebank dataset from %s' % test_npz_path)
        return (train_data['x'], train_data['y']), \
               (valid_data['x'], valid_data['y']), \
               (test_data['x'], test_data['y'])


    train = load_stanford_treebank_dataset_file(STANFORD_TREEBANK_TRAIN)
    valid = load_stanford_treebank_dataset_file(STANFORD_TREEBANK_VALID)
    test = load_stanford_treebank_dataset_file(STANFORD_TREEBANK_TEST)

    train = word_2_vecs(train[0], train[1])
    valid = word_2_vecs(valid[0], valid[1])
    test = word_2_vecs(test[0], test[1])

    np.savez(train_npz_path, x=train[0], y=train[1])
    np.savez(valid_npz_path, x=valid[0], y=valid[1])
    np.savez(test_npz_path, x=test[0], y=test[1])

    print('saving standford imdb train dataset to %s' % train_npz_path)
    print('saving standford imdb valid dataset to %s' % valid_npz_path)
    print('saving standford imdb test dataset to %s' % test_npz_path)

    return train, valid, test


def load_stanford_treebank_dataset_file(path):
    dataset = pytreebank.import_tree_corpus(path)

    labels = list()
    sentences = list()

    for tree in dataset:
        for label, sentence in tree.to_labeled_lines():
            sentences.append(sentence)
            if label >= 2:
                labels.append(1) # positive
            else:
                labels.append(0) # negative

    return sentences, np.array(labels)


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
        x_pos_train, y_pos_train = word_2_vecs(x_pos_train, y_pos_train)
        x_neg_train, y_neg_train = word_2_vecs(x_neg_train, y_neg_train)
        x_pos_test, y_pos_test = word_2_vecs(x_pos_test, y_pos_test)
        x_neg_test, x_neg_test = word_2_vecs(x_neg_test, x_neg_test)
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
    print('saving standford imdb test dataset to %s' % test_npz_path)

    return (x_train, y_train), (x_test, y_test)


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
        x, y = word_2_vecs(sentences, y)
    else:
        x = skip_thoughts_vecs(sentences)

    np.savez(npz_path, x=x, y=y)
    print('saved uci dataset to %s' % npz_path)

    return x, y

def small_proj_dataset(vec_rep=WORD2VEC):
    if vec_rep == WORD2VEC:
        npz_path = W2V_SMALL_NPZ
    else:
        npz_path = SKP_SMALL_NPZ

    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        print('loaded small dataset from %s' % npz_path)
        return data['x_train'], data['y_train'], data['x_test'], data['y_test']

    x, y = load_uci_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.33,
                                                        random_state=42)

    print('saved small to %s' % npz_path)
    np.savez(npz_path, x_train=x_train, y_train=y_train, x_test=x_test,
            y_test=y_test)

    return x_train, y_train, x_test, y_test

def proj_data(vec_rep=WORD2VEC):

    if vec_rep == WORD2VEC:
        npz_path = W2V_ALL_NPZ
    else:
        npz_path = SKP_ALL_NPZ

    if os.path.isfile(npz_path):
        data = np.load(npz_path)
        print('loaded dataset from %s' % npz_path)
        return (data['x_train'], data['y_train']), \
                (data['x_valid'], data['y_valid']), \
                (data['x_test'], data['y_test'])

    uci_x, uci_y = load_uci_dataset()
    train, test = load_stanford_imdb_dataset(vec_rep)

    stanford_x_train = train[0]
    stanford_y_train = train[1]
    stanford_x_test = test[0]
    stanford_y_test = test[1]

    x, y = merge_datasets(uci_x, uci_y, stanford_x_train, stanford_y_train)
    x, y = merge_datasets(x, y, stanford_x_test, stanford_y_test)


    x_train, x_valid, y_train, y_valid = train_test_split(x,
                                                        y,
                                                        test_size=0.35,
                                                        random_state=42)

    x_valid, x_test, y_valid, y_test = train_test_split(x_valid,
                                                        y_valid,
                                                        test_size=0.45,
                                                        random_state=42)

    print('saved all to %s' % npz_path)
    np.savez(npz_path,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            x_test=x_test,
            y_test=y_test)

    return (x_train, y_train), \
            (x_valid, y_valid), \
            (x_test, y_test)

if __name__ == '__main__':
    load_stanford_imdb_dataset(SKIPTHOUGHTS)

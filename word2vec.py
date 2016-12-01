import os, io
import numpy as np
import pytreebank
from sklearn.model_selection import train_test_split


# Label 1 (for positive), or
#       0 (for negative)

PROJECT_DATASET_FILE = './datasets/proj_as_vectors.npz'

STANFORD_DATASET_PATH   = './datasets/aclImdb'
STANFORD_TRAIN_POS_PATH = './datasets/aclImdb/train_pos_vectors.npz'
STANFORD_TRAIN_NEG_PATH = './datasets/aclImdb/train_neg_vectors.npz'
STANFORD_TEST_POS_PATH  = './datasets/aclImdb/test_pos_vectors.npz'
STANFORD_TEST_NEG_PATH  = './datasets/aclImdb/test_neg_vectors.npz'

STANFORD_TREEBANK_TEST     = './datasets/stanford_sentiment_treebank/trees/test.txt'
STANFORD_TREEBANK_TEST_NPZ = './datasets/stanford_sentiment_treebank/test.npz'

UCI_DATASET_PATH    = './datasets/sentiment_labelled_sentences'
UCI_IMDB_FILEPATH   = os.path.join(UCI_DATASET_PATH, 'imdb_labelled.txt')
UCI_YELP_FILEPATH   = os.path.join(UCI_DATASET_PATH, 'yelp_labelled.txt')
UCI_AMAZON_FILEPATH = os.path.join(UCI_DATASET_PATH, 'amazon_cells_labelled.txt')
UCI_NP_W2V_FILEPATH = os.path.join(UCI_DATASET_PATH, 'dataset_as_w2v.npz')

# global nlp bag of magic
nlp = None

def init_nlp_packages():
    import spacy
    from textacy.preprocess      import preprocess_text, normalize_whitespace
    from gensim.models.word2vec  import Word2Vec, LineSentence

    global nlp
    nlp = spacy.load('en')

def clean_sentence(sentences):
    c = sentences.replace('-', ' ') # people use to concatinate words
    c = normalize_whitespace(c)
    c = preprocess_text(c,
                        lowercase=True,
                        no_numbers=True,
                        no_punct=True,
                        no_contractions=True)
    return c

def stanford_treebank_test_dataset(save_to_file=True, load_if_exists=True):
    if load_if_exists and os.path.isfile(STANFORD_TREEBANK_TEST_NPZ):
        data = np.load(STANFORD_TREEBANK_TEST_NPZ)
        print('loaded stanford test treebank dataset from %s' % STANFORD_TREEBANK_TEST_NPZ)
        return (data['x'], data['y'])

    dataset = pytreebank.import_tree_corpus(STANFORD_TREEBANK_TEST)
    example = dataset[0]

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

    if save_to_file or not os.path.isfile(STANFORD_TREEBANK_TEST_NPZ):
        print('saving stanford treebank test dataset to %s' %
                STANFORD_TREEBANK_TEST_NPZ)
        np.savez(STANFORD_TREEBANK_TEST_NPZ, x=x, y=y)

def uci_dataset_from_file(filepath):
    inputs = list()
    labels = list()
    with open(filepath) as f:
        for line in f:
            line_list = line.split()
            label     = int(line_list.pop(-1))
            sentence  = ' '.join(line_list)

            inputs.append(sentence)
            labels.append(label)

    return (inputs, labels)

def load_uci_sentiment_dataset():
    imdb = uci_dataset_from_file(UCI_IMDB_FILEPATH)
    yelp = uci_dataset_from_file(UCI_YELP_FILEPATH)
    amzn = uci_dataset_from_file(UCI_AMAZON_FILEPATH)

    return (imdb, yelp, amzn)

def uci_dataset_as_vectors(save_to_file=True, load_if_exists=True):
    if load_if_exists and os.path.isfile(UCI_NP_W2V_FILEPATH):
        data = np.load(UCI_NP_W2V_FILEPATH)
        print('loaded uci dataset from %s' % UCI_NP_W2V_FILEPATH)
        return (data['x'], data['y'])

    imdb, yelp, amzn = load_uci_sentiment_dataset()

    sentences = imdb[0] + yelp[0] + amzn[0]
    x = np.zeros((len(sentences), 300))
    y = np.array(imdb[1] + yelp[1] + amzn[1])

    for idx, s in enumerate(sentences):
        doc    = nlp(s)
        x[idx] = doc.vector


    if save_to_file:
        print('saving uci dataset to %s' % UCI_NP_W2V_FILEPATH)
        np.savez(UCI_NP_W2V_FILEPATH, x=x, y=y)

    return (x, y)

def train_uci_word2vec():
    imdb, yelp, amzn = load_uci_sentiment_dataset()


    x = imdb[0] + yelp[0] + amzn[0]
    y = imdb[1] + yelp[1] + amzn[1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    stringio = io.StringIO('\n'.join(x_train))
    line_sentences = LineSentence(stringio)

    print("Training")
    w2v = Word2Vec(line_sentences, workers=4, min_count=1)

    return w2v


def read_stanford_dir(path, saved_model_path, load_if_exists=True):
    if load_if_exists and os.path.isfile(saved_model_path):
        data = np.load(saved_model_path)
        print('loaded stanford dataset from %s' % saved_model_path)
        return (data['x'], data['y'])

    x = np.zeros((12500, 300))
    y = np.zeros((12500,))
    i = 0
    for f in os.listdir(path):
        if f.endswith('.txt'):
            # try:
            filepath = os.path.join(path, f)
            with open(filepath) as f:
                sentence = f.read()
            _, label = os.path.basename(filepath).replace('.txt', '').split('_')
            x[i] = nlp(sentence).vector
            if int(label) > 5:
                y[i] = 1 # positive
            i += 1
            print('++++ DONE %s' % os.path.basename(filepath))
            # except Exception as e:
            #     print('Failed to read file %s' % filepath)

    print('saving of %s dataset to %s' % (path, saved_model_path))
    np.savez(saved_model_path, x=x, y=y)

    return (x, y)


def stanford_dataset_as_vectors():
    train_pos_dir = os.path.join(STANFORD_DATASET_PATH, 'train', 'pos')
    train_neg_dir = os.path.join(STANFORD_DATASET_PATH, 'train', 'neg')
    test_pos_dir  = os.path.join(STANFORD_DATASET_PATH, 'test', 'pos')
    test_neg_dir  = os.path.join(STANFORD_DATASET_PATH, 'test', 'neg')

    x_pos_train, y_pos_train = read_stanford_dir(train_pos_dir,
                                                STANFORD_TRAIN_POS_PATH)
    x_neg_train, y_neg_train = read_stanford_dir(train_neg_dir,
                                                STANFORD_TRAIN_NEG_PATH)
    x_pos_test, y_pos_test   = read_stanford_dir(test_pos_dir,
                                                STANFORD_TEST_POS_PATH)
    x_neg_test, y_neg_test   = read_stanford_dir(test_neg_dir,
                                                STANFORD_TEST_NEG_PATH)

    x = np.zeros((50000, 300))
    x[:12500]      = x_pos_train
    x[12500:25000] = x_neg_train
    x[25000:37500] = x_pos_test
    x[37500:50000] = x_neg_test

    y = np.zeros((50000, ))
    y[:12500]      = y_pos_train
    y[12500:25000] = y_neg_train
    y[25000:37500] = y_pos_test
    y[37500:50000] = y_neg_test

    return (x, y)

def project_dataset_as_vectors(override=False):
    if not override and os.path.isfile(PROJECT_DATASET_FILE):
        data = np.load(PROJECT_DATASET_FILE)
        return (data['x_train'], data['x_test'], data['y_train'], data['y_test'])


    x1, y1 = uci_dataset_as_vectors()

    x2, y2 = stanford_dataset_as_vectors()

    n = x1.shape[0] + x2.shape[0]
    x_size = (n, 300)

    x = np.zeros((n, 300))
    y = np.zeros(n)

    x[:x1.shape[0]] = x1
    x[x1.shape[0]:] = x2

    y[:y1.shape[0]] = y1
    y[y1.shape[0]:] = y2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    if override or not os.path.isfile(PROJECT_DATASET_FILE):
        np.savez(PROJECT_DATASET_FILE,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test)

    return (x_train, x_test, y_train, y_test)

def main():
    init_nlp_packages()
    # uci_wv = uci_dataset_as_vectors()
    # stanford_wv = stanford_dataset_as_vectors()
    # dataset = project_dataset_as_vectors(override=True)
    dataset = stanford_treebank_test_dataset()
    _start_shell(locals())

if __name__ == '__main__':
    main()

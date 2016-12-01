import tensorflow as tf
import numpy as np
import spacy

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def predict_sentence(w_var, b_var):
    nlp = spacy.load('en')
    X = tf.placeholder('float64', (1, 300))
    pred = tf.nn.sigmoid(tf.matmul(X, w_var) + b_var)

    while True:
        s = input('Enter sentence: ').strip()
        if s == '0':
            print('bye!!')
            break
        v = nlp(s).vector.reshape((1, 300)).astype('float64')
        p = pred.eval({X: v}).flatten()[0]
        if p >= 0.50:
            print('+++')
        else:
            print('---')


def classify(model_path, interactive=True):
    saver = tf.train.import_meta_graph(model_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        all_vars = tf.trainable_variables()
        w_var = all_vars[0]
        b_var = all_vars[1]
        predict_sentence(w_var, b_var)


def main():
    classify('lr_model.ckpt')

if __name__ == '__main__':
    main()

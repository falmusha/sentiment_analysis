import tensorflow as tf
import numpy as np

from util import hot_encode_2_classes, load_stanford_treebank_dataset
from word2vec import uci_dataset_as_vectors

INPUT_DIM = 300
CLASS_NUM = 2

def tensors(x, y, learning_rate=0.01):
    # Set model weights
    with tf.name_scope('weights'):
        w = tf.Variable(tf.random_normal((INPUT_DIM, CLASS_NUM)), name='weight')

    # Construct a linear model
    with tf.name_scope('biases'):
        b = tf.Variable(tf.random_normal([CLASS_NUM]), name='bias')

    with tf.name_scope('sigmoid_prediction'):
        # pred = tf.nn.sigmoid(tf.matmul(x, w) + b, name='pred')
        pred = tf.nn.softmax(tf.matmul(x, w) + b) # Softmax


    # Minimize error
    with tf.name_scope('cost_l2_loss'):
        # cost = tf.nn.l2_loss(pred-y, name='cost')
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))


    # Gradient Descent
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    return pred, cost, optimizer


def train(train_x, train_y, test_x, test_y, epochs=10, batch_size=100):
    number_of_smaples = train_x.shape[0]

    # tf Graph Input
    with tf.name_scope('input'):
        X = tf.placeholder('float', (None, INPUT_DIM), name='x-input')
        Y = tf.placeholder('float', (None, CLASS_NUM), name='y-input')

    pred, cost, optimizer = tensors(X, Y)

    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init_op)

        # Fit all training data
        for epoch in range(epochs):
            avg_cost = 0.
            total_batches = int(train_x.shape[0] / batch_size)

            for i in range(total_batches):
                start = i * batch_size
                end   = start + batch_size
                x_batch = train_x[start:end]
                y_batch = train_y[start:end]

                _, c = sess.run([optimizer, cost],
                                feed_dict={X: x_batch, Y: y_batch})

                avg_cost += c / total_batches
                    # Display logs per epoch step

            if (epoch+1) % 1 == 0:
                print('Epoch:', '%04d' % (epoch+1),
                      "cost=",
                      '{:.9f}'.format(avg_cost))

        print('Optimization Finished!')

        # Test model
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.round(pred), Y)
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print('Accuracy:', accuracy.eval({X: test_x, Y: test_y}))
        save_path = saver.save(sess, "./lr_model_treebank.ckpt")

def main():
    dataset = load_stanford_treebank_dataset()
    x = dataset[0]
    y = hot_encode_2_classes(dataset[1])

    train(x[:354103], y[:354103], x[354103:], y[354103:], 1000, 1000)

if __name__ == '__main__':
    main()

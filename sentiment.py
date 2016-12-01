import tensorflow as tf
import numpy as np

from word2vec import uci_dataset_as_vectors, project_dataset_as_vectors, \
stanford_treebank_test_dataset


def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
        user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def evaluate_test(model_path, test_X, test_Y):
    saver = tf.train.import_meta_graph(model_path + '.meta')

    X = tf.placeholder('float64', (None, 300))
    Y = tf.placeholder('float64', (None, 1))

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        all_vars = tf.trainable_variables()
        w_var = all_vars[0]
        b_var = all_vars[1]
        pred = tf.nn.sigmoid(tf.matmul(X, w_var) + b_var)
        correct_prediction = tf.equal(tf.round(pred), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('TEST Accuracy:', accuracy.eval({X: test_X, Y: test_Y}))

def lr_tensors(input_dim, learning_rate=0.0001):
    # N number of examples (batch_size)
    # M dimensionality of input
    N, M = input_dim

    # tf Graph Input
    with tf.name_scope('input'):
        X = tf.placeholder('float64', (None, M), name='x-input')
        Y = tf.placeholder('float64', (None, 1), name='y-input') # binary

    # Set model weights
    with tf.name_scope('weights'):
        W = tf.Variable(np.random.randn(M, 1), name='weight', dtype='float64')

    # Construct a linear model
    with tf.name_scope('biases'):
        b = tf.Variable(np.random.randn(1), name='bias', dtype='float64')

    with tf.name_scope('sigmoid_prediction'):
        pred = tf.nn.sigmoid(tf.matmul(X, W) + b)

    # Minimize error
    with tf.name_scope('cost_l2_loss'):
        cost = tf.nn.l2_loss(pred-Y, name='cost')

    # Gradient Descent
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    return pred, cost, optimizer, X, Y

def train_lr(data, epochs, batch_size=1000,  display_step=10):

    train_X, test_X, train_Y, test_Y = data

    s_learning_rate = tf.train.exponential_decay(learning_rate=0.0008,
                                                global_step=1,
                                                decay_steps=train_X.shape[0],
                                                decay_rate=0.95,
                                                staircase=True)

    train_Y = train_Y.astype('float64').reshape((train_Y.shape[0], 1))
    test_Y  = test_Y.astype('float64').reshape((test_Y.shape[0], 1))

    pred, cost, optimizer, X, Y = lr_tensors(train_X.shape, s_learning_rate)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.round(pred), Y)
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.histogram_summary('pred_summary', pred)
    tf.histogram_summary('y_summary', Y)
    tf.scalar_summary('cost_summary', cost)
    tf.scalar_summary('accuracy_summary', accuracy)

    summary_op = tf.merge_all_summaries()

    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init_op)

        writer = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())

        # Fit all training data
        for epoch in range(epochs):
            avg_cost = 0.
            total_batches = int(train_X.shape[0] / batch_size)

            for i in range(total_batches):
                # print('batch %d/%d' % (i, total_batches))
                start = i * batch_size
                end   = start + batch_size
                x_batch = train_X[start:end]
                y_batch = train_Y[start:end]

                _, c, summary = sess.run([optimizer, cost, summary_op],
                                feed_dict={X: x_batch, Y: y_batch})
                writer.add_summary(summary, epoch * total_batches + i)

                avg_cost += c / total_batches
                    # Display logs per epoch step

            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        print('Accuracy:', accuracy.eval({X: test_X, Y: test_Y}))
        save_path = saver.save(sess, "./lr_model.ckpt")
        print("Model saved in file: %s" % save_path)


def main():
    # train_lr(project_dataset_as_vectors(), 10000)
    x, y = stanford_treebank_test_dataset()
    evaluate_test('./lr_model.ckpt', x, y.reshape(y.shape[0], 1))

if __name__ == '__main__':
    main()

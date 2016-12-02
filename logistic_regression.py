import os
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from util import hot_encode_2_classes, load_stanford_treebank_dataset, small_proj_dataset
from word2vec import uci_dataset_as_vectors

CLASS_NUM = 2

def tensors(x, y, input_dim, learning_rate=0.001):
    # Set model weights
    w = tf.Variable(tf.random_normal((input_dim, CLASS_NUM)), name='Weights')

    # Construct a linear model
    b = tf.Variable(tf.random_normal([CLASS_NUM]), name='Bias')

    with tf.name_scope('Model'):
        pred = tf.nn.sigmoid(tf.matmul(x, w) + b)

    # Minimize error
    with tf.name_scope('Loss'):
        cost = tf.nn.l2_loss(pred-y)

    # Gradient Descent
    with tf.name_scope('Optimizer'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

    return pred, cost, optimizer


def train(training_set, validation_set, test_set, model_name,
        epochs=10,
        batch_size=100,
        learning_rate=0.001):

    display_step = 10
    train_x, train_y = training_set[0], training_set[1]
    valid_x, valid_y = validation_set[0], validation_set[1]
    test_x, test_y = test_set[0], test_set[1]

    number_of_smaples = train_x.shape[0]

    INPUT_DIM = train_x.shape[1]

    X = tf.placeholder('float', (None, INPUT_DIM), name='input')
    Y = tf.placeholder('float', (None, CLASS_NUM), name='label')

    pred, cost, optimizer = tensors(X, Y, INPUT_DIM, learning_rate)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.round(pred), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver()

    training_cost_summary = tf.scalar_summary('training_loss', cost)
    training_accuracy_summary = tf.scalar_summary('training_accuracy', accuracy)
    training_merged_sum = tf.merge_summary([training_accuracy_summary, training_cost_summary])

    validation_cost_summary = tf.scalar_summary('validation_loss', cost)
    validation_accuracy_summary = tf.scalar_summary('validation_accuracy', accuracy)
    validation_merged_sum = tf.merge_summary([validation_accuracy_summary,
        validation_cost_summary])

    logs_name = '%s_epochs_%d_batch_size_%d_lrate_%f' \
            % (model_name, epochs, batch_size, learning_rate)
    logs_path = os.path.join('logs', logs_name)

    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init_op)

        summary_writer = tf.train.SummaryWriter(logs_path,
                graph=tf.get_default_graph())

        best_acc = 0.0
        best_epoch = 0
        # Fit all training data
        for epoch in range(epochs):
            training_avg_cost = 0.
            total_batches = int(train_x.shape[0] / batch_size)

            for i in range(total_batches):
                start = i * batch_size
                end   = start + batch_size
                x_batch = train_x[start:end]
                y_batch = train_y[start:end]

                _, training_c, training_summary = \
                        sess.run([optimizer, cost, training_merged_sum],
                                feed_dict={X: x_batch, Y: y_batch})

                summary_writer.add_summary(training_summary, epoch * total_batches + i)
                avg_cost += training_c / total_batches

            validation_c, validation_summary = \
                    sess.run([cost, validation_merged_sum],
                            feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_summary(validation_summary, epoch)

            epoch_acc = float(accuracy.eval({x: test_x, y: test_y}))
            if round(epoch_acc, 2) > round(best_acc, 2):
                best_acc = epoch_acc
                best_epoch = epoch
                print('BEST IS %f' % best_acc)
                best_model_name = \
                '%s_accuracy_%s_epochs_%d_batch_size_%d_lrate_%f_h1_%d_h2_%d_h3_%d_best_at_epoch_%d' \
                        % (best_acc, model_name, epochs, batch_size, learning_rate, n_hidden_1, \
                        n_hidden_2, n_hidden_3, epoch)
                best_save_path = \
                        saver.save(sess, os.path.join('models', best_model_name))

            if (epoch+1) % 10 == 0:
                print('Epoch:', '%04d' % (epoch+1),
                      "cost=",
                      '{:.9f}'.format(avg_cost))

        print('Optimization Finished!')

        acc = accuracy.eval({X: test_x, Y: test_y})
        print('Accuracy:', acc)

        model_name = './%s_accuracy_%f_epochs_%d_batch_size_%d_lrate_%f.ckpt'\
                % (model_name, acc, epochs, batch_size, learning_rate) 

        save_path = saver.save(sess, model_name)

def main():
    x_train, y_train, x_test, y_test = small_proj_dataset()
    y_train = hot_encode_2_classes(y_train)
    y_test = hot_encode_2_classes(y_test)

    train(x_train, y_train, x_test, y_test, 'lr_small_w2v_with_Adadelta', 15000, 500, 0.5)
    tf.reset_default_graph()
    train(x_train, y_train, x_test, y_test, 'lr_small_w2v_with_Adadelta', 15000, 500, 0.1)
    tf.reset_default_graph()
    train(x_train, y_train, x_test, y_test, 'lr_small_w2v_with_Adadelta', 15000, 500, 0.01)
    tf.reset_default_graph()
    train(x_train, y_train, x_test, y_test, 'lr_small_w2v_with_Adadelta', 15000, 500, 0.001)
    tf.reset_default_graph()
    train(x_train, y_train, x_test, y_test, 'lr_small_w2v_with_Adadelta', 15000, 500, 0.0001)
    tf.reset_default_graph()
    train(x_train, y_train, x_test, y_test, 'lr_small_w2v_with_Adadelta', 15000, 500, 0.00001)
    tf.reset_default_graph()

if __name__ == '__main__':
    main()

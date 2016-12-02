import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from word2vec import uci_dataset_as_vectors
from util import hot_encode_2_classes, load_stanford_treebank_dataset, \
small_proj_dataset, proj_data

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    return out_layer


# Parameters
def train(training_set, validation_set, test_set, model_name,
        epochs=10,
        batch_size=100,
        learning_rate=0.001,
        n_hidden_1=256,
        n_hidden_2=256,
        n_hidden_3=256):
    display_step = 10
    train_x, train_y = training_set[0], training_set[1]
    valid_x, valid_y = validation_set[0], validation_set[1]
    test_x, test_y = test_set[0], test_set[1]

    print('NN TRAINING: lr %f - %d x %d x %d' \
            % (learning_rate, n_hidden_1, n_hidden_2, n_hidden_3))

    n_input = train_x.shape[1]
    n_classes = 2

    x = tf.placeholder("float", [None, n_input], name='input')
    y = tf.placeholder("float", [None, n_classes], name='label')

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    with tf.name_scope('Model'):
        pred = multilayer_perceptron(x, weights, biases)

    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initializing the variables
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    training_cost_summary = tf.scalar_summary('training_loss', cost)
    training_accuracy_summary = tf.scalar_summary('training_accuracy', accuracy)
    training_merged_sum = tf.merge_summary([training_accuracy_summary, training_cost_summary])

    validation_cost_summary = tf.scalar_summary('validation_loss', cost)
    validation_accuracy_summary = tf.scalar_summary('validation_accuracy', accuracy)
    validation_merged_sum = tf.merge_summary([validation_accuracy_summary,
        validation_cost_summary])

    logs_name = '%s_epochs_%d_batch_size_%d_lrate_%f_h1_%d_h2_%d_h3_%d' \
            % (model_name, epochs, batch_size, learning_rate, n_hidden_1, \
            n_hidden_2, n_hidden_3)
    logs_path = os.path.join('logs', logs_name)

    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)

    with tf.Session() as sess:
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(logs_path,
                graph=tf.get_default_graph())

        best_acc = 0.0
        best_epoch = 0
        # Training cycle
        for epoch in range(epochs):
            training_avg_cost = 0.
            total_batch = int(train_x.shape[0]/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                start = i * batch_size
                end   = start + batch_size
                batch_x, batch_y = train_x[start:end], train_y[start:end]

                _, training_c, training_summary = \
                        sess.run([optimizer, cost, training_merged_sum],
                                feed_dict={x: batch_x, y: batch_y})
                summary_writer.add_summary(training_summary, epoch * total_batch + i)

                # Compute average loss
                training_avg_cost += training_c / total_batch

            validation_c, validation_summary = \
                    sess.run([cost, validation_merged_sum],
                            feed_dict={x: valid_x, y: valid_y})

            summary_writer.add_summary(validation_summary, epoch * total_batch)
            epoch_acc = float(accuracy.eval({x: valid_x, y: valid_y}))

            if round(epoch_acc, 2) > round(best_acc, 2):
                best_acc = epoch_acc
                best_epoch = epoch
                print('BEST VALIDATION ACCURACY IS %f' % best_acc)
                best_model_name = \
                '%s_accuracy_%s_epochs_%d_batch_size_%d_lrate_%f_h1_%d_h2_%d_h3_%d' \
                        % (best_acc, model_name, epochs, batch_size, learning_rate, n_hidden_1, \
                        n_hidden_2, n_hidden_3)
                best_save_path = \
                        saver.save(sess, os.path.join('models', best_model_name))

            if epoch % display_step == 0:
                print('Epoch:', '%04d' % (epoch+1), \
                    'Training   cost =', '{:.9f}'.format(training_avg_cost), \
                    'Validation cost =', '{:.9f}'.format(validation_c))

                if epoch - best_epoch > 400:
                    print('EARLY STOPPING at epoch %d' % epoch)
                    break


        print('Optimization Finished!')
        train_acc = accuracy.eval({x: train_x, y: train_y})
        print('Accuracy TRAIN: ', train_acc)
        valid_acc = accuracy.eval({x: valid_x, y: valid_y})
        print('Accuracy VALID: ', valid_acc)
        test_acc = accuracy.eval({x: test_x, y: test_y})
        print('Accuracy  TEST: ', test_acc)

def main():
    train_set, valid_set, test_set = proj_data()
    x_train = train_set[0]
    y_train = hot_encode_2_classes(train_set[1])
    x_valid = valid_set[0]
    y_valid = hot_encode_2_classes(valid_set[1])
    x_test = test_set[0]
    y_test = hot_encode_2_classes(test_set[1])

    training_set, validation_set, test_set = (x_train, y_train), \
                                             (x_valid, y_valid), \
                                             (x_test, y_test)

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.1)
    tf.reset_default_graph()

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.01)
    tf.reset_default_graph()

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.001)
    tf.reset_default_graph()

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.1,
            512, 512, 512)
    tf.reset_default_graph()

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.01,
            512, 512, 512)
    tf.reset_default_graph()

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.001,
            512, 512, 512)
    tf.reset_default_graph()

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.1,
            800, 800, 800)
    tf.reset_default_graph()

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.01,
            800, 800, 800)
    tf.reset_default_graph()

    train(training_set, validation_set, test_set, 'nn_all_w2v_adam',
            3000,
            1000,
            0.001,
            800, 800, 800)
    tf.reset_default_graph()

if __name__ == '__main__':
    main()

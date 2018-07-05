import numpy as np 
import tensorflow as tf 
from read_data import *
import math

batch_size = 64
seq_max_len = 930
nb_label = 3
embedding_size = 100
nb_linear_inside = 100
nb_lstm_inside = 100
layers = 1
TRAINING_ITERATIONS = 2000
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.00001
flag_train = True
accuracies = []
data, label, mask, sequence_length, weight_mask, protvec, key_aa = prepare_data("../input", "../output")
data = np.asarray(data[20 * len(data) / 100 :])
label = np.asarray(label[20 * len(label) / 100 :])
mask = np.asarray(mask[20 * len(mask) / 100 :])
sequence_length = np.asarray(sequence_length[20 * len(sequence_length) / 100 :])
weight_mask = np.asarray(weight_mask[20 * len(weight_mask) / 100 :])
vocabulary_size = len(protvec)

graph = tf.Graph()
with graph.as_default():
  tf_X = tf.placeholder(tf.int64, shape=[None, seq_max_len])
  tf_y = tf.placeholder(tf.int64, shape=[None, seq_max_len])
  tf_word_embeddings = tf.placeholder(tf.float32, shape=[vocabulary_size, embedding_size])
  tf_X_binary_mask = tf.placeholder(tf.float32, shape=[None, seq_max_len])
  tf_weight_mask = tf.placeholder(tf.float32, shape=[None, seq_max_len])
  tf_seq_len = tf.placeholder(tf.int64, shape=[None, ])
  keep_prob = tf.placeholder(tf.float32)
  
  ln_w = tf.Variable(tf.truncated_normal([embedding_size, nb_linear_inside], stddev=1.0 / math.sqrt(embedding_size)))
  ln_b = tf.Variable(tf.zeros([nb_linear_inside]))
  sent_w = tf.Variable(tf.truncated_normal([nb_lstm_inside, nb_label],
                       stddev=1.0 / math.sqrt(2 * nb_lstm_inside)))
  sent_b = tf.Variable(tf.zeros([nb_label]))
  y_labels = tf.one_hot(tf_y, nb_label,
              on_value = 1.0,
              off_value = 0.0,
              axis = -1)
  X_train = tf.nn.embedding_lookup(tf_word_embeddings, tf_X)
  X_train = tf.transpose(X_train, [1, 0, 2])
  # Reshaping to (n_steps*batch_size, n_input)
  X_train = tf.reshape(X_train, [-1, embedding_size])
  X_train = tf.add(tf.matmul(X_train, ln_w), ln_b)
  X_train = tf.nn.relu(X_train)
  X_train = tf.split(axis=0, num_or_size_splits=seq_max_len, value=X_train)
  # Creating the forward and backwards cells
  # X_train = tf.stack(X_train)
  lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(nb_lstm_inside, forget_bias=0.5)
  lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(nb_lstm_inside, forget_bias=0.5)
  outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                         lstm_bw_cell,
                         X_train,
                         #sequence_length=tf_seq_len,
                         dtype='float32')
  output_fw, output_bw = tf.split(outputs, [nb_lstm_inside, nb_lstm_inside], 2)
  structure = tf.reshape(tf.add(output_fw, output_bw), [-1, nb_lstm_inside]) 
  structure = tf.nn.dropout(structure, keep_prob)
  structure = tf.add(tf.matmul(structure, sent_w), sent_b)
  structure = tf.split(axis=0, num_or_size_splits=seq_max_len, value=structure)
  # Change back dimension to [batch_size, n_step, n_input]
  structure = tf.stack(structure)
  structure = tf.transpose(structure, [1, 0, 2])
  structure = tf.multiply(structure, tf.expand_dims(tf_X_binary_mask, 2))
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=structure, labels=y_labels))
  regularization = WEIGHT_DECAY * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name or "Bias" in tf_var.name))
  prediction = tf.argmax(tf.nn.softmax(structure), 2)
  correct_prediction = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(prediction, tf_y), tf.float32), tf_X_binary_mask))
  optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy + regularization)
  saver = tf.train.Saver()

# Training & testing
with tf.Session(graph=graph) as session:
  for re in range(5):
    train_data = np.asarray([data[i] for i in range(len(data)) if i % 5 != re])
    train_label = np.asarray([label[i] for i in range(len(data)) if i % 5 != re])
    test_data = np.asarray([data[i] for i in range(len(data)) if i % 5 == re])
    test_label = np.asarray([label[i] for i in range(len(data)) if i % 5 == re])
    train_binary_mask = np.asarray([mask[i] for i in range(len(data)) if i % 5 != re])
    test_binary_mask = np.asarray([mask[i] for i in range(len(data)) if i % 5 == re])
    train_seq_len = np.asarray([sequence_length[i] for i in range(len(data)) if i % 5 != re])
    test_seq_len = np.asarray([sequence_length[i] for i in range(len(data)) if i % 5 == re])
    train_weight_mask = np.asarray([weight_mask[i] for i in range(len(data)) if i % 5 != re])
    accuracy_tmp = 0
    nb_sample_train = len(train_data)
    # Modeling
    session.run(tf.global_variables_initializer())
    for it in range(TRAINING_ITERATIONS):
      if (it * batch_size % nb_sample_train + batch_size < nb_sample_train):
        index = it * batch_size % nb_sample_train
      else:
        index = nb_sample_train - batch_size
      _, correct_prediction_train, cost_train = session.run([optimizer, correct_prediction, cross_entropy], 
                            feed_dict={tf_X: train_data[index : index + batch_size],
                                      tf_y: train_label[index : index + batch_size],
                                      tf_X_binary_mask: train_binary_mask[index : index + batch_size],
                                      tf_word_embeddings: protvec,
                                      tf_seq_len: train_seq_len[index : index + batch_size],
                                      tf_weight_mask: train_weight_mask[index: index + batch_size],
                                      keep_prob: 0.5})
      print('training_accuracy => %.3f, cost value => %.5f for step %d' %(float(correct_prediction_train)/np.sum(train_binary_mask[index : index + batch_size]), cost_train, it))
      
      if it % 50 == 0:
        correct_prediction_test, test_prediction = session.run([correct_prediction, prediction], 
                          feed_dict={tf_X: test_data,
                                    tf_y: test_label,
                                    tf_X_binary_mask: test_binary_mask,
                                    tf_word_embeddings: protvec,
                                    tf_seq_len: test_seq_len,
                                    keep_prob: 1.0})

        y_test = []
        y_pred = []
        for i in range(len(test_seq_len)):
          y_test.extend(test_label[i][0 : test_seq_len[i]])
          y_pred.extend(test_prediction[i][0 : test_seq_len[i]])
        cm = confusion_matrix(y_test, y_pred)
        
        print(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        print('test accuracy => %.3f' %(float(correct_prediction_test)/np.sum(test_binary_mask)))
        if float(correct_prediction_test)/np.sum(test_binary_mask) > accuracy_tmp:
          accuracy_tmp = float(correct_prediction_test)/np.sum(test_binary_mask)

    accuracies.append(accuracy_tmp)
    print accuracies

print accuracies
print np.mean(accuracies)
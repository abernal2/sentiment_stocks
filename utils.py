#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:01:47 2019

@author: Armando Bernal
"""

import re
import string
import pandas as pd
from collections import Counter
import numpy as np
import tensorflow as tf

def preprocess_ST_message(text):
    """
    Preprocesses raw message data for analysis
    :param text: String. ST Message
    :return: List of Strings.  List of processed text tokes
    """
    # Define ST Regex Patters
    REGEX_PRICE_SIGN = re.compile(r'\$(?!\d*\.?\d+%)\d*\.?\d+|(?!\d*\.?\d+%)\d*\.?\d+\$')
    REGEX_PRICE_NOSIGN = re.compile(r'(?!\d*\.?\d+%)(?!\d*\.?\d+k)\d*\.?\d+')
    REGEX_TICKER = re.compile('\$[a-zA-Z]+')
    REGEX_USER = re.compile('\@\w+')
    REGEX_LINK = re.compile('https?:\/\/[^\s]+')
    REGEX_HTML_ENTITY = re.compile('\&\w+')
    REGEX_NON_ACSII = re.compile('[^\x00-\x7f]')
    REGEX_PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation.replace('<', '')).replace('>', ''))
    REGEX_NUMBER = re.compile(r'[-+]?[0-9]+')

    text = text.lower()

    # Replace ST "entitites" with a unique token
    text = re.sub(REGEX_TICKER, ' <TICKER> ', text)
    text = re.sub(REGEX_USER, ' <USER> ', text)
    text = re.sub(REGEX_LINK, ' <LINK> ', text)
    text = re.sub(REGEX_PRICE_SIGN, ' <PRICE> ', text)
    text = re.sub(REGEX_PRICE_NOSIGN, ' <NUMBER> ', text)
    text = re.sub(REGEX_NUMBER, ' <NUMBER> ', text)
    # Remove extraneous text data
    text = re.sub(REGEX_HTML_ENTITY, "", text)
    text = re.sub(REGEX_NON_ACSII, "", text)
    text = re.sub(REGEX_PUNCTUATION, "", text)
    # Tokenize and remove < and > that are not in special tokens
    words = " ".join(token.replace("<", "").replace(">", "")
                     if token not in ['<TICKER>', '<USER>', '<LINK>', '<PRICE>', '<NUMBER>']
                     else token
                     for token
                     in text.split())

    return words

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict maps a vocab word to and integeter
             The second maps an integer back to to the vocab word
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab, 1)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def encode_ST_messages(messages, vocab_to_int):
    """
    Encode ST Sentiment Labels
    :param messages: list of list of strings. List of message tokens
    :param vocab_to_int: mapping of vocab to idx
    :return: list of ints. Lists of encoded messages
    """
    messages_encoded = []
    for message in messages:
        messages_encoded.append([vocab_to_int[word] for word in message.split()])

    return np.array(messages_encoded)

def encode_ST_labels(labels):
    """
    Encode ST Sentiment Labels
    :param labels: Input list of labels
    :return: numpy array.  The encoded labels
    """
    return np.array([1 if sentiment == 'bullish' else 0 for sentiment in labels])

def drop_empty_messages(messages, labels):
    """
    Drop messages that are left empty after preprocessing
    :param messages: list of encoded messages
    :return: tuple of arrays. First array is non-empty messages, second array is non-empty labels
    """
    non_zero_idx = [ii for ii, message in enumerate(messages) if len(message) != 0]
    messages_non_zero = np.array([messages[ii] for ii in non_zero_idx])
    labels_non_zero = np.array([labels[ii] for ii in non_zero_idx])
    return messages_non_zero, labels_non_zero

def zero_pad_messages(messages, seq_len):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param seq_ken: Input int, maximum sequence input length
    :return: numpy array.  The encoded labels
    """
    messages_padded = np.zeros((len(messages), seq_len), dtype=int)
    for i, row in enumerate(messages):
        messages_padded[i, -len(row):] = np.array(row)[:seq_len]

    return np.array(messages_padded)

def train_val_test_split(messages, labels, split_frac, random_seed=None):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param labels: Input list of encoded labels
    :param split_frac: Input float, training split percentage
    :return: tuple of arrays train_x, val_x, test_x, train_y, val_y, test_y
    """
    # make sure that number of messages and labels allign
    assert len(messages) == len(labels)
    # random shuffle data
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(messages))
    messages_shuf = np.array(messages)[shuf_idx] 
    labels_shuf = np.array(labels)[shuf_idx]

    #make splits
    split_idx = int(len(messages_shuf)*split_frac)
    train_x, val_x = messages_shuf[:split_idx], messages_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(val_x)*0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    return train_x, val_x, test_x, train_y, val_y, test_y
    
def get_batches(x, y, batch_size=100):
    """
    Batch Generator for Training
    :param x: Input array of x data
    :param y: Input array of y data
    :param batch_size: Input int, size of batch
    :return: generator that returns a tuple of our x batch and y batch
    """
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        


def retrain_network(model_dir, batch_size, train_x, train_y, val_x, val_y, epochs):
    
    inputs_, labels_, keep_prob_ = model_inputs()
    embed = build_embedding_layer(inputs_, vocab_size, embed_size)
    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate)
    accuracy = build_accuracy(predictions, labels_)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        n_batches = len(train_x)//batch_size
        for e in range(epochs):
            state = sess.run(initial_state)
            
            train_acc = []
            for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob_: keep_prob,
                        initial_state: state}
                loss_, state, _,  batch_acc = sess.run([loss, final_state, optimizer, accuracy], feed_dict=feed)
                train_acc.append(batch_acc)
                
                if (ii + 1) % n_batches == 0:
                    
                    val_acc = []
                    val_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
                    for xx, yy in get_batches(val_x, val_y, batch_size):
                        feed = {inputs_: xx,
                                labels_: yy[:, None],
                                keep_prob_: 1,
                                initial_state: val_state}
                        val_batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(val_batch_acc)
                    
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Batch: {}/{}...".format(ii+1, n_batches),
                          "Train Loss: {:.3f}...".format(loss_),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Accuracy: {:.3f}".format(np.mean(val_acc)))
    
        saver.save(sess, "checkpoints/sentiment2.ckpt")

def model_inputs():
    """
    Create the model inputs
    """
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs_, labels_, keep_prob_

def build_embedding_layer(inputs_, vocab_size, embed_size):
    """
    Create the embedding layer
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)
    
    return embed

def build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size):
    """
    Create the LSTM layers
    """
    lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]
    # Add dropout to the cell
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
    
    return initial_state, lstm_outputs, cell, final_state

def build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate):
    """
    Create the Loss function and Optimizer
    """
    predictions = tf.contrib.layers.fully_connected(lstm_outputs[:, -1], 1, activation_fn=tf.sigmoid)
    loss = tf.losses.mean_squared_error(labels_, predictions)
    optimzer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    
    return predictions, loss, optimzer

def build_accuracy(predictions, labels_):
    """
    Create accuracy
    """
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return accuracy

def build_and_train_network(lstm_sizes, vocab_size, embed_size, epochs, batch_size,
                            learning_rate, keep_prob, train_x, val_x, train_y, val_y):
    
    inputs_, labels_, keep_prob_ = model_inputs()
    embed = build_embedding_layer(inputs_, vocab_size, embed_size)
    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate)
    accuracy = build_accuracy(predictions, labels_)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        n_batches = len(train_x)//batch_size
        for e in range(epochs):
            state = sess.run(initial_state)
            
            train_acc = []
            for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob_: keep_prob,
                        initial_state: state}
                loss_, state, _,  batch_acc = sess.run([loss, final_state, optimizer, accuracy], feed_dict=feed)
                train_acc.append(batch_acc)
                
                if (ii + 1) % n_batches == 0:
                    
                    val_acc = []
                    val_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
                    for xx, yy in get_batches(val_x, val_y, batch_size):
                        feed = {inputs_: xx,
                                labels_: yy[:, None],
                                keep_prob_: 1,
                                initial_state: val_state}
                        val_batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(val_batch_acc)
                    
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Batch: {}/{}...".format(ii+1, n_batches),
                          "Train Loss: {:.3f}...".format(loss_),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Accuracy: {:.3f}".format(np.mean(val_acc)))
    
        saver.save(sess, "checkpoints/sentiment.ckpt")
        
def test_network(model_dir, batch_size, test_x, test_y):
    
    inputs_, labels_, keep_prob_ = model_inputs()
    embed = build_embedding_layer(inputs_, vocab_size, embed_size)
    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate)
    accuracy = build_accuracy(predictions, labels_)
    
    saver = tf.train.Saver()
    
    test_acc = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        test_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
        for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob_: 1,
                    initial_state: test_state}
            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test Accuracy: {:.3f}".format(np.mean(test_acc)))
        
        
# read data from csv file
data = pd.read_csv("StockTwits_SPY_Sentiment_2017.csv",
                   encoding="utf-8",
                   index_col=0)

# get messages and sentiment labels
messages = data.message.values
labels = data.sentiment.values

# View sample of messages with sentiment

for i in range(10):
    print("Messages: {}...".format(messages[i]),
          "Sentiment: {}".format(labels[i]))
    
messages = np.array([preprocess_ST_message(message) for message in messages])
full_lexicon = " ".join(messages).split()
vocab_to_int, int_to_vocab = create_lookup_tables(full_lexicon)
messages_lens = Counter([len(x) for x in messages])
print("Zero-length messages: {}".format(messages_lens[0]))
print("Maximum message length: {}".format(max(messages_lens)))
print("Average message length: {}".format(np.mean([len(x) for x in messages])))
messages, labels = drop_empty_messages(messages, labels)
messages = encode_ST_messages(messages, vocab_to_int)
labels = encode_ST_labels(labels)
messages = zero_pad_messages(messages, seq_len=244)

train_x, val_x, test_x, train_y, val_y, test_y = train_val_test_split(messages, labels, split_frac=0.80)

lstm_sizes = [128, 64]
vocab_size = len(vocab_to_int) + 1 #add one for padding
embed_size = 300
epochs = 10
batch_size = 256
learning_rate = 0.15
keep_prob = 0.5

# Training
with tf.Graph().as_default():
    build_and_train_network(lstm_sizes, vocab_size, embed_size, epochs, batch_size,
                            learning_rate, keep_prob, train_x, val_x, train_y, val_y)
    
# Testing
with tf.Graph().as_default():
    test_network('checkpoints', batch_size, test_x, test_y)

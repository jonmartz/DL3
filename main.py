import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import librosa.display
import spacy
# import pretty_midi
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


def get_vocabulary_and_word_embeddings(texts_train, texts_test, nlp):
    # todo: maybe remove the '&' from all texts
    word_indexes = {}
    vocabulary, word_embeddings = [], []
    for texts in [texts_train, texts_test]:
        for text in texts:
            tokens = nlp(text.strip())
            for token in tokens:
                word = token.text
                if word not in vocabulary:
                    word_indexes[word] = len(word_indexes)
                    vocabulary.append(word)
                    word_embeddings.append(token.vector)
    return word_indexes, np.array(word_embeddings), np.array(vocabulary)


def get_train_and_test_sets(texts_train, texts_test, word_indexes, nlp):
    x_train, y_train, x_test, y_test = [], [], [], []
    for x, y, texts in [[x_train, y_train, texts_train], [x_test, y_test, texts_test]]:
        for text in texts:
            text_as_indexes = [word_indexes[token.text] for token in nlp(text.strip())]
            x.append(text_as_indexes[:-1])
            y.append(text_as_indexes[1:])
    return x_train, y_train, x_test, y_test


def get_data_train(x_train, y_train):
    x_tensors = tuple((tf.constant(i) for i in x_train))
    y_tensors = tuple((tf.constant(i) for i in y_train))
    return tf.data.Dataset.from_tensors((x_tensors, y_tensors))


def build_model(vocabulary_size, word_embeddings, layer_sizes):
    model = Sequential(name='RNN')
    model.add(layers.Embedding(vocabulary_size,
                               word_embeddings.shape[1],
                               embeddings_initializer=Constant(word_embeddings),
                               trainable=False))
    for layer_size in layer_sizes:
        model.add(layers.LSTM(layer_size,
                              return_sequences=True,
                              # stateful=True,
                              ))
    model.add(layers.Dense(vocabulary_size))
    model.compile(optimizer='adam', loss=loss)
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def get_checkpoints_callback(checkpoint_dir):
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)


def generate_text(model, start_string, word_indexes, vocabulary, text_len=30):
    # Evaluation step (generating text using the learned model)

    # Converting our start string to numbers (vectorizing)
    # input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims([word_indexes[start_string]], 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(text_len):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(vocabulary[predicted_id])

    return (start_string + ' ' + ' '.join(text_generated))


def write_to_file(history, generated_texts, layer_sizes, iteration):
    layers_str = '_'.join([str(i) for i in layer_sizes])
    dir_name = 'drive/My Drive/Colab results/lyric generator results/layers_%s/%d' % (layers_str, iteration)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    to_file = []
    header = "epoch"
    scores = []
    epoch = 0
    for key in history:
        header = "%s,%s" % (header, key)
        scores.append(history[key])

    to_file.append(header)
    for i in range(len(scores[0])):
        epoch += 1
        line = "%d" % epoch
        for j in range(len(scores)):
            line = "%s,%s" % (line, scores[j][i])
        to_file.append(line)

    with open('%s/train_results.csv' % dir_name, 'w') as file:
        for line in to_file:
            file.write("%s\n" % line)

    with open('%s/test_results.csv' % dir_name, 'w') as file:
        for line in generated_texts:
            file.write("%s\n" % line)


# def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
#     # Use librosa's specshow function for displaying the piano roll
#     librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
#                              hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
#                              fmin=pretty_midi.note_number_to_hz(start_pitch))
#
#
# def pretty_midi_test():
#     pm = pretty_midi.PrettyMIDI('dataset/midi_files/2_Unlimited_-_Get_Ready_for_This.mid')
#
#     # plt.figure(figsize=(12, 4))
#     # plot_piano_roll(pm, 24, 84)
#     # plt.show()
#
#     # Let's look at what's in this MIDI file
#     print('There are {} time signature changes'.format(len(pm.time_signature_changes)))
#     print('There are {} instruments'.format(len(pm.instruments)))
#     print('Instrument 3 has {} notes'.format(len(pm.instruments[0].notes)))
#     print('Instrument 4 has {} pitch bends'.format(len(pm.instruments[4].pitch_bends)))
#     print('Instrument 5 has {} control changes'.format(len(pm.instruments[5].control_changes)))


# dataset_dir = 'dataset'  # for pc
dataset_dir = 'DL3/dataset'  # for colab
texts_train = pd.read_csv('%s/lyrics_train_set.csv' % dataset_dir, header=None)[2]
texts_test = pd.read_csv('%s/lyrics_test_set.csv' % dataset_dir, header=None)[2]

print('building vocabulary...')
nlp = spacy.load('en_core_web_md')  # 300 dim embeddings
# nlp = spacy.load('en_core_web_sm')  # smaller embeddings
word_indexes, word_embeddings, vocabulary = get_vocabulary_and_word_embeddings(texts_train, texts_test, nlp)

print('building train and test sets...')
x_train, y_train, x_test, y_test = get_train_and_test_sets(texts_train, texts_test, word_indexes, nlp)
data_train = get_data_train(x_train, y_train)

# build model
layer_sizes = [128]
model = build_model(len(vocabulary), word_embeddings, layer_sizes)
model.summary()
checkpoint_dir = './training_checkpoints'
checkpoint_callback = get_checkpoints_callback(checkpoint_dir)

print('training model...')
history = model.fit(data_train, epochs=100, callbacks=[checkpoint_callback])

# get best checkpoint
tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# generate text for all the 5 test melodies, 3 times
for iteration in range(3):
    generated_texts = []
    for i, text in enumerate(x_test):
        print('\ntarget text %d:' % i)
        print(' '.join(vocabulary[text]))

        generated_text = generate_text(model, vocabulary[text[0]], word_indexes, vocabulary, text_len=len(text))
        print('generated text %d:' % i)
        print(generated_text)

        generated_texts.append(generated_text)
    write_to_file(history.history, generated_texts, layer_sizes, iteration + 1)

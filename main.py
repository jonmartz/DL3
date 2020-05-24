import os
import sys
import numpy as np
import pandas as pd
import spacy
import pretty_midi
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant


def get_midis_and_texts(instances, dataset_dir, instrument_indexes):
    midis = []
    texts = []
    failed_num = 0
    for index, row in instances.iterrows():
        artist = row[0].strip().replace(' ', '_')
        song_name = row[1].strip().replace(' ', '_')
        filename = '%s_-_%s.mid' % (artist, song_name)
        failed_text = '\t'
        try:
            midi = pretty_midi.PrettyMIDI('%s/midi_files/%s' % (dataset_dir, filename))
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    instrument_id = instrument.program
                    if instrument_id not in instrument_indexes:
                        instrument_indexes[instrument_id] = len(instrument_indexes)
            midis.append(midi)
            texts.append(row[2])
        except:
            failed_text = ' FAILED'
            failed_num += 1
        print('%d/%d%s\t%s' % (index + 1, len(instances), failed_text, filename))
    print('FAILED = %d/%d' % (failed_num, len(instances)))
    return midis, texts


def get_midi_vectors(midis, texts, instrument_indexes, mode):
    midi_vectors = []
    for midi, text in zip(midis, texts):
        general_tempo = midi.estimate_tempo()
        general_chroma = midi.get_chroma()
        total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
        instruments_presence = [0] * len(instrument_indexes)  # one entry per instrument
        total_velocity = sum(sum(general_chroma))
        general_semitones = [sum(semitone) / total_velocity for semitone in general_chroma]
        instruments_semitones = [0] * (len(instrument_indexes) * 12)  # one entry per instrument per semitone
        for instrument in midi.instruments:
            if not instrument.is_drum:
                instrument_index = instrument_indexes[instrument.program]  # instrument's index in the midi vector
                instrument_presence = len(instrument.notes) / total_notes
                instruments_presence[instrument_index] = instrument_presence
                instrument_chroma = instrument.get_chroma()
                total_velocity = sum(sum(instrument_chroma))
                instrument_semitones = [sum(semitone) / total_velocity for semitone in instrument_chroma]
                start_index = 12 * instrument_index  # jump to the instrument's semitones section in the midi vector
                instruments_semitones[start_index: start_index + 12] = instrument_semitones
        midi_vector = [general_tempo] + instruments_presence + general_semitones + instruments_semitones
        midi_vectors.append(midi_vector)
    return midi_vectors


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
            text_word_indexes = [word_indexes[token.text] for token in nlp(text.strip())]
            x.append(text_word_indexes[:-1])
            y.append(text_word_indexes[1:])
    return x_train, y_train, x_test, y_test


def get_train_and_val_data(x_train, y_train, val_split):
    val_len = int(len(x_train) * val_split)
    x_train_tensors = tuple((tf.constant(i) for i in x_train[:-val_len]))
    y_train_tensors = tuple((tf.constant(i) for i in y_train[:-val_len]))
    x_val_tensors = tuple((tf.constant(i) for i in x_train[-val_len:]))
    y_val_tensors = tuple((tf.constant(i) for i in y_train[-val_len:]))
    train_data = tf.data.Dataset.from_tensors((x_train_tensors, y_train_tensors))
    val_data = tf.data.Dataset.from_tensors((x_val_tensors, y_val_tensors))
    return train_data, val_data


def build_model(vocabulary_size, word_embeddings, layer_sizes):
    model = Sequential(name='RNN')
    model.add(layers.Embedding(vocabulary_size, word_embeddings.shape[1], trainable=False,
                               embeddings_initializer=Constant(word_embeddings)))
    for layer_size in layer_sizes:
        model.add(layers.LSTM(layer_size, return_sequences=True))
    model.add(layers.Dense(vocabulary_size))
    model.compile(optimizer='adam', loss=loss)
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def get_checkpoints_callback(checkpoint_dir):
    filepath_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    return tf.keras.callbacks.ModelCheckpoint(filepath=filepath_prefix, monitor='val_loss', save_weights_only=True)


def generate_text(model, input_word, word_indexes, vocabulary, text_len, temperature=1.0):
    """

    :param model:
    :param input_word:
    :param word_indexes:
    :param vocabulary:
    :param text_len:
    :param temperature: low values result -> predictable text, high values -> surprising text.
    :return:
    """
    current_word_index = tf.expand_dims([word_indexes[input_word]], 0)
    text_generated = []
    model.reset_states()
    for i in range(text_len):
        predicted_words = tf.squeeze(model(current_word_index), 0) / temperature  # remove the batch dimension
        # use a categorical distribution to predict the next word
        predicted_id = tf.random.categorical(predicted_words, num_samples=1)[-1, 0].numpy()
        text_generated.append(vocabulary[predicted_id])
        current_word_index = tf.expand_dims([predicted_id], 0)

    return '%s %s' % (input_word, ' '.join(text_generated))


def write_to_file(history, generated_texts, layer_sizes, iteration):
    layers_str = '_'.join([str(i) for i in layer_sizes])
    dir_name = 'drive/My Drive/Colab results/lyric generator results/layers_%s' % layers_str
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

    with open('%s/test_results_%d.csv' % (dir_name, iteration), 'w') as file:
        for line in generated_texts:
            file.write("%s\n" % line)


mode = 'complete song'
# mode = 'sliced song'
dataset_dir = 'dataset'  # for pc
# dataset_dir = 'DL3/dataset'  # for colab
instances_train = pd.read_csv('%s/lyrics_train_set.csv' % dataset_dir, header=None)
instances_test = pd.read_csv('%s/lyrics_test_set.csv' % dataset_dir, header=None)
instrument_indexes = {}
print('loading midis and texts...')
midis_train, texts_train = get_midis_and_texts(instances_train, dataset_dir, instrument_indexes)
midis_test, texts_test = get_midis_and_texts(instances_test, dataset_dir, instrument_indexes)
print('generating midi vectors...')
midi_vectors_train = get_midi_vectors(midis_train, texts_train, instrument_indexes, mode)
midi_vectors_test = get_midi_vectors(midis_test, midis_test, instrument_indexes, mode)

print('building vocabulary...')
# nlp = spacy.load('en_core_web_md')  # 300 dim embeddings
nlp = spacy.load('en_core_web_sm')  # smaller embeddings
word_indexes, word_embeddings, vocabulary = get_vocabulary_and_word_embeddings(texts_train, texts_test, nlp)

print('building train and test sets...')
val_split = 0.2
x_train, y_train, x_test, y_test = get_train_and_test_sets(texts_train, texts_test, word_indexes, nlp)
train_data, val_data = get_train_and_val_data(x_train, y_train, val_split)

# build model
layer_sizes = [128]
model = build_model(len(vocabulary), word_embeddings, layer_sizes)
model.summary()
checkpoint_dir = './training_checkpoints'
checkpoint_callback = get_checkpoints_callback(checkpoint_dir)

print('training model...')
history = model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[checkpoint_callback])

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

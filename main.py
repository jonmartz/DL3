import os
import sys
import numpy as np
import pandas as pd
import spacy
import pretty_midi
import json


# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt

# from keras.initializers import Constant
# from keras.layers import LSTM, Dense, concatenate, Input, Embedding
# from keras.models import Model, load_model
# from keras.losses import CategoricalCrossentropy
# from keras.callbacks import ModelCheckpoint


def get_midis_and_texts(instances, dataset_dir, instrument_indexes, set):
    cache_path = '%s/texts_%s.json' % (cache_dir, set)
    if not os.path.exists(cache_path):
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
                    # if not instrument.is_drum:
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
        with open(cache_path, 'w') as file:
            json.dump(texts, file)
    else:
        with open(cache_path, 'r') as file:
            texts = json.load(file)
        midis = None
    return midis, texts


def get_vocabulary_and_word_embeddings(texts_train, texts_test, nlp):
    if not os.path.exists('%s/word_indexes.json' % cache_dir):
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
                        word_embeddings.append(token.vector.tolist())
        for file_name in ['word_indexes', 'word_embeddings', 'vocabulary']:
            with open('%s/%s.json' % (cache_dir, file_name), 'w') as file:
                json.dump(eval(file_name), file)
    else:
        structures = []
        for file_name in ['word_indexes', 'word_embeddings', 'vocabulary']:
            with open('%s/%s.json' % (cache_dir, file_name), 'r') as file:
                structures.append(json.load(file))
        word_indexes, word_embeddings, vocabulary = structures
    return word_indexes, np.array(word_embeddings), np.array(vocabulary)


def get_midi_embeddings():
    if not os.path.exists('%s/midi_embeddings.json' % cache_dir):
        print('train:')
        midi_vectors_train, midi_vectors_sliced_train, slice_indexes_train = get_midi_vectors(
            midis_train, instrument_indexes, texts_train)
        print('test:')
        midi_vectors_test, midi_vectors_sliced_test, slice_indexes_test = get_midi_vectors(
            midis_test, instrument_indexes, texts_test, init_slice_index=slice_indexes_train[-1][-1] + 1)
        # midi_embeddings = np.concatenate([midi_vectors_train, midi_vectors_test])
        # midi_sliced_embeddings = np.concatenate([midi_vectors_sliced_train, midi_vectors_sliced_test])
        midi_embeddings = midi_vectors_train + midi_vectors_test
        midi_sliced_embeddings = midi_vectors_sliced_train + midi_vectors_sliced_test

        for file_name in ['midi_embeddings', 'midi_sliced_embeddings', 'slice_indexes_train', 'slice_indexes_test']:
            with open('%s/%s.json' % (cache_dir, file_name), 'w') as file:
                json.dump(eval(file_name), file)
    else:
        structures = []
        for file_name in ['midi_embeddings', 'midi_sliced_embeddings', 'slice_indexes_train', 'slice_indexes_test']:
            with open('%s/%s.json' % (cache_dir, file_name), 'r') as file:
                structures.append(json.load(file))
        midi_embeddings, midi_sliced_embeddings, slice_indexes_train, slice_indexes_test = structures
    return np.array(midi_embeddings), np.array(midi_sliced_embeddings), slice_indexes_train, slice_indexes_test


def get_midi_vectors(midis, instrument_indexes, texts, init_slice_index=0):
    num_phrases_by_song = [text.count('&') for text in texts]
    midi_vectors = []
    midi_vectors_sliced = []
    slice_indexes = []  # index of midi slice embeddings
    for i, midi in enumerate(midis):
        print('%d/%d' % (i + 1, len(midis)))
        midi_tempo = midi.estimate_tempo()
        midi_chroma = midi.get_chroma()
        instruments_presence = get_slice_instruments_presence(midi, 1, midi.get_end_time(), instrument_indexes)[0]
        total_velocity = sum(sum(midi_chroma))
        midi_semitones = [sum(semitone) / total_velocity for semitone in midi_chroma]
        midi_vector = [midi_tempo] + instruments_presence + midi_semitones
        midi_vectors.append(midi_vector)
        for midi_vector_slice in get_midi_vector_slices(midi, num_phrases_by_song[i], instrument_indexes):
            midi_vectors_sliced.append(midi_vector_slice)
        if i == 0:
            slice_indexes.append(list(range(num_phrases_by_song[i] + init_slice_index)))
        else:
            next_index = slice_indexes[-1][-1] + 1
            slice_indexes.append(list(range(next_index, next_index + num_phrases_by_song[i])))
    return midi_vectors, midi_vectors_sliced, slice_indexes


def get_midi_vector_slices(midi, num_slices, instrument_indexes):
    midi_end_time = midi.get_end_time()
    slice_len = midi_end_time / num_slices
    slice_starts = [i * slice_len for i in range(num_slices)]
    slice_ends = slice_starts[1:] + [midi_end_time]

    slice_tempos = get_slice_tempos(midi, slice_starts, slice_ends, midi_end_time)
    slice_semitones = get_slice_semitones(midi, num_slices)
    slice_instruments_presence = get_slice_instruments_presence(midi, num_slices, midi_end_time, instrument_indexes)

    midi_vector_slices = []
    for i in range(num_slices):
        midi_vector_slice = [slice_tempos[i]] + slice_instruments_presence[i] + slice_semitones[i]
        midi_vector_slices.append(midi_vector_slice)
    return midi_vector_slices


def get_slice_tempos(midi, slice_starts, slice_ends, midi_end_time):
    tempo_starts, tempos = midi.get_tempo_changes()
    tempo_starts, tempos = list(tempo_starts), list(tempos)
    tempo_ends = tempo_starts[1:] + [midi_end_time]  # for weighted average of tempos inside same slice
    slice_tempos = [[] for i in range(len(slice_starts))]  # tempos in each slice
    slice_tempo_lens = [[] for i in range(len(slice_starts))]  # len of tempos in each slice
    slice_id = 0
    for tempo_start, tempo_end, tempo in zip(tempo_starts, tempo_ends, tempos):
        while slice_ends[slice_id] < tempo_end:  # next change is outside of slice
            slice_tempos[slice_id].append(tempo)
            slice_tempo_start = max(slice_starts[slice_id], tempo_start)
            slice_tempo_lens[slice_id].append(slice_ends[slice_id] - slice_tempo_start)
            slice_id += 1
        slice_tempos[slice_id].append(tempo)
        slice_tempo_start = max(slice_starts[slice_id], tempo_start)
        slice_tempo_lens[slice_id].append(tempo_end - slice_tempo_start)
    return [np.average(i, weights=j) for i, j in zip(slice_tempos, slice_tempo_lens)]


def get_slice_semitones(midi, num_slices):
    slices_chromas = np.array_split(midi.get_chroma(), num_slices, axis=1)
    slice_semitones = []
    for slice_chroma in slices_chromas:
        slice_velocity = sum(sum(slice_chroma))
        if slice_velocity == 0:
            slice_semitones.append([0] * len(slice_chroma))
        else:
            slice_semitones.append([sum(semitone) / slice_velocity for semitone in slice_chroma])
    return slice_semitones


def get_slice_instruments_presence(midi, num_slices, midi_end_time, instrument_indexes):
    midi_note_matrix = get_note_matrix(midi, midi_end_time, instrument_indexes)
    slices_note_matrix = np.array_split(midi_note_matrix, num_slices, axis=1)
    slice_instruments_presence = []
    for slice_note_matrix in slices_note_matrix:
        total_presence = np.sum(slice_note_matrix)
        if total_presence == 0:
            slice_instruments_presence.append([0] * len(instrument_indexes))
        else:
            slice_instruments_presence.append((np.sum(slice_note_matrix, axis=1) / total_presence).tolist())
    return slice_instruments_presence


def get_note_matrix(midi, midi_end_time, instrument_indexes):
    note_matrix = np.zeros([len(instrument_indexes), int(midi_end_time)])
    for instrument in midi.instruments:
        row = note_matrix[instrument_indexes[instrument.program]]
        for note in instrument.notes:
            row[int(note.start): int(note.end) + 1] = 1
    return note_matrix


def get_set(texts, midi_indexes, slices_indexes, word_indexes, nlp):
    x_words, x_midis, x_midis_sliced, y = [], [], [], []
    for i in range(len(texts)):
        text, midi_index = texts[i], midi_indexes[i]
        words = [token.text for token in nlp(text.strip())]
        text_word_indexes = [word_indexes[word] for word in words]  # indexes of words in this text
        x_words.extend(text_word_indexes[:-1])  # shape (len(text)-1 = num of "words in text except the last one")
        x_midis.extend([midi_index] * (len(text_word_indexes) - 1))  # shape (len(text)-1, len(midi_vector))
        x_midis_sliced.extend(get_x_midi_sliced(words[:-1], slices_indexes[i]))
        y.extend(text_word_indexes[1:])
    return [np.array(x_words), np.array(x_midis), np.array(x_midis_sliced)], np.array(y)


def get_x_midi_sliced(words, slice_indexes):
    x_midi_sliced = []
    midi_slice_idx = 0
    for word in words:
        x_midi_sliced.append(slice_indexes[midi_slice_idx])
        if word == '&':  # end of phrase
            midi_slice_idx += 1
    return x_midi_sliced


def build_model(word_embeddings, midi_embeddings, lstm_lens, dense_lens, complex_mode=False, slice_embeddings=None):
    # word component
    word_in = Input(shape=(1,), name='word_in')
    lstm_word = Embedding(word_embeddings.shape[0], word_embeddings.shape[1],
                          trainable=False, weights=[word_embeddings], input_length=1)(word_in)
    for lstm_len in lstm_lens:
        lstm_word = LSTM(lstm_len)(lstm_word)
    # midi component
    midi_in = Input(shape=(1,), name='midi_in')
    midi_embedding = Embedding(midi_embeddings.shape[0], midi_embeddings.shape[1],
                        trainable=False, weights=[midi_embeddings], input_length=1)(midi_in)
    midi_embedding = Reshape(tuple([midi_embedding.shape[2]]))(midi_embedding)
    # concatenation
    if complex_mode:
        # midi slice component
        slice_in = Input(shape=(1,), name='slice_in')
        lstm_slice = Embedding(slice_embeddings.shape[0], slice_embeddings.shape[1],
                               trainable=False, weights=[slice_embeddings], input_length=1)(slice_in)
        for lstm_len in lstm_lens:
            lstm_slice = LSTM(lstm_len)(lstm_slice)
        dense = concatenate([lstm_word, midi_embedding, lstm_slice])
    else:
        dense = concatenate([lstm_word, midi_embedding])
    # dense layers
    for dense_len in dense_lens:
        dense = Dense(dense_len)(dense)
    output = Dense(word_embeddings.shape[0], activation='softmax')(dense)
    if complex_mode:
        model = Model([word_in, midi_in, slice_in], output, name='complex')
    else:
        model = Model([word_in, midi_in], output, name='simple')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


def generate_text(model, input_word, word_indexes, vocabulary, text_len, midi, temperature=1.0):
    # current_word_index = tf.expand_dims([word_indexes[input_word]], 0)
    current_word_index = word_indexes[input_word]
    text_generated = []
    model.reset_states()
    for i in range(text_len):
        # predicted_words = tf.squeeze(model.predict([[current_word_index], [midi]]), 0) / temperature  # remove the batch dimension
        predicted_words = model.predict([[current_word_index], [midi]])
        # use a categorical distribution to predict the next word
        predicted_id = tf.random.categorical(predicted_words, num_samples=1)[-1, 0].numpy()
        text_generated.append(vocabulary[predicted_id])
        # print(text_generated[-1])
        current_word_index = predicted_id
    return '%s %s' % (input_word, ' '.join(text_generated))


def generate_texts(model, x_test, vocabulary):
    mode = model.name
    generated_texts = []
    prev_midi_idx = -1
    for i in range(len(x_test[0])):
        curr_midi_idx = x_test[1][i]
        if mode == 'complex':
            midi_slice_idx = x_test[2][i]
        if prev_midi_idx != curr_midi_idx:  # got to next song
            prev_midi_idx = curr_midi_idx
            if i != 0:  # save text generated from last song
                generated_texts.append(' '.join(generated_text))
            print('test song %d' % (len(generated_texts) + 1))
            first_word_idx = x_test[0][i]  # take first word
            word_idx = first_word_idx
            generated_text = [vocabulary[word_idx]]
            model.reset_states()
        if mode == 'no midi':
            pred_word_idxs = model.predict([[word_idx]])
        elif mode == 'simple':
            pred_word_idxs = model.predict([[word_idx], [curr_midi_idx]])
        else:
            pred_word_idxs = model.predict([[word_idx], [curr_midi_idx], [midi_slice_idx]])
        word_idx = tf.random.categorical(pred_word_idxs, num_samples=1)[-1, 0].numpy()
        generated_text.append(vocabulary[word_idx])
    generated_texts.append(' '.join(generated_text))  # append last generated text
    return generated_texts


def write_to_file(history, generated_texts, layer_sizes, iteration, model_name):
    layers_str = '_'.join([str(i) for i in layer_sizes])
    dir_name = 'drive/My Drive/Colab results/lyric generator results/layers_%s/%s' % (layers_str, model_name)
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


# base_dir = 'DL3/'  # for colab
base_dir = ''
train_subset = 0  # for speeding up debugging. select 0 for whole train set

dataset_dir = '%sdataset' % base_dir
cache_dir = '%scaches/%d' % (base_dir, train_subset)
instances_train = pd.read_csv('%s/lyrics_train_set.csv' % dataset_dir, header=None)
instances_test = pd.read_csv('%s/lyrics_test_set.csv' % dataset_dir, header=None)

if train_subset > 0:
    instances_train = instances_train[:train_subset]

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
instrument_indexes = {}
print('loading texts and midis...')
print('train:')
midis_train, texts_train = get_midis_and_texts(instances_train, dataset_dir, instrument_indexes, 'train')
print('test:')
midis_test, texts_test = get_midis_and_texts(instances_test, dataset_dir, instrument_indexes, 'test')

print('building word embeddings...')
nlp = spacy.load('en_core_web_md')  # 300 dim embeddings
# nlp = spacy.load('en_core_web_sm')  # smaller embeddings
word_indexes, word_embeddings, vocabulary = get_vocabulary_and_word_embeddings(texts_train, texts_test, nlp)

print('building midi embeddings...')
midi_embeddings, midi_sliced_embeddings, slice_indexes_train, slice_indexes_test = get_midi_embeddings()

# val_split = 0.2
# complex_mode = True
#
# train_len = len(texts_train)
# midi_indexes_train = list(range(train_len))
# midi_indexes_test = list(range(train_len, train_len + len(texts_test)))
# val_len = int(len(texts_train) * val_split)
# print('building train set...')
# x_train, y_train = get_set(texts_train[:-val_len], midi_indexes_train[:-val_len], slice_indexes_train[:-val_len],
#                            word_indexes, nlp, complex_mode)
# print('building validation set...')
# x_val, y_val = get_set(texts_train[-val_len:], midi_indexes_train[-val_len:], slice_indexes_train[-val_len:],
#                        word_indexes, nlp, complex_mode)
# print('building test set...')
# x_test, y_test = get_set(texts_test, midi_indexes_test, slice_indexes_test, word_indexes, nlp, complex_mode)
#
# # build model
# lstm_lens = [128]
# dense_lens = [1024]
# epochs = 10
# batch_size = 256
#
# model = build_model(word_embeddings, midi_embeddings, lstm_lens, dense_lens, complex_mode, midi_sliced_embeddings)
# model.summary()
# checkpoint = ModelCheckpoint('checkpoint.h5', save_best_only=True)
# history = model.fit(x_train, y_train, batch_size, epochs, callbacks=[checkpoint], validation_data=(x_val, y_val))
# model = load_model('model2_checkpoint.h5')
#
# # generate text for all the 5 test melodies, 3 times
# for iteration in range(3):
#     print('iteration %d/3' % (iteration + 1))
#     generated_texts = generate_texts(model, x_test, vocabulary)
#     write_to_file(history.history, generated_texts, lstm_lens, iteration + 1, model.name)
#
# print('done')

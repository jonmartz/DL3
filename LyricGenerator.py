import os
import sys
import numpy as np
import pandas as pd
import spacy
import pretty_midi
import json
import tensorflow as tf
from keras.layers import LSTM, Dense, concatenate, Input, Embedding, Dropout
from keras.models import Model, load_model
from keras.losses import CategoricalCrossentropy
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Reshape
from numpy.random import choice


def get_midis_and_texts(instances, dataset_dir, instrument_indexes, set):
    """
    Extract the midis and texts from the train or test set files
    :param instances: dataframe with the song names and lyrics
    :param dataset_dir: directory of the dataset
    :param instrument_indexes: dict to be updated during this function as new instruments are encountered
    :param set: string = "train" or "test" that indicates what set to extract
    :return: [raw midis in set, lyrics for all songs in set]
    """
    cache_path = '%s/texts_%s.json' % (cache_dir, set)
    if not os.path.exists(cache_path):  # if cache doesn't exist
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
                    instrument_id = instrument.program
                    if instrument_id not in instrument_indexes:
                        instrument_indexes[instrument_id] = len(instrument_indexes)
                midis.append(midi)
                texts.append(row[2])
            except:  # if midi is not readable
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
    """
    Process the texts to get the word embeddings and vocabulary of the whole dataset
    :param texts_train: lyrics of the train set
    :param texts_test: lyrics of the test set
    :param nlp: the Spacy model
    :return: [dict where word->index, np.array where index->word embedding, np.array where index->word]
    """
    if not os.path.exists('%s/word_indexes.json' % cache_dir):  # if cache doesn't exist
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
    """
    Extract the embedding vectors of the whole and slices midis
    :return: [np.array index->midi embedding, np.array midi_sliced_embeddings index->slice embedding,
                "list where each entry i is a list with the slice indexes for midi i in train set", "same for test set"]
    """
    if not os.path.exists('%s/midi_embeddings.json' % cache_dir):  # if cache doesn't exist
        print('train:')
        midi_vectors_train, midi_vectors_sliced_train, slice_indexes_train = get_midi_vectors(
            midis_train, instrument_indexes, texts_train)
        print('test:')
        midi_vectors_test, midi_vectors_sliced_test, slice_indexes_test = get_midi_vectors(
            midis_test, instrument_indexes, texts_test, init_slice_index=slice_indexes_train[-1][-1] + 1)
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
    """
    Extract the embedding vectors of the whole and slices midis in the train or test set
    :param midis: of train or test set
    :param instrument_indexes: computed by the method "get_midis_and_texts"
    :param texts: lyrics of the train or test set
    :param init_slice_index: initial slice index – only if sending test set will not be zero
    :return: [np.array of midi embeddings, np.array of midi slice embedding, slice_indexes as in get_midi_embeddings]
    """
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
    """
    Extract slice embedding vectors for a midi file
    :param midi: raw midi to be sliced
    :param num_slices: number of equal length slices to split the midi into
    :param instrument_indexes: computed by the method "get_midis_and_texts"
    :return: np.array with embeddings of the midi slices
    """
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
    """
    Compute average tempo in each slice, weighted by the amount of time each tempo occurs in the slice
    :param midi: to slice
    :param slice_starts: start times of slices
    :param slice_ends: end times of slices
    :param midi_end_time: time where midi ends
    :return: weighted average of tempos in each slice
    """
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
    """
    Compute average of each one of the 12 semitones in each midi
    :param midi: to slice
    :param num_slices: to split the midi into
    :return: a list with the average semitones (np.array of size 12) for each slice
    """
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
    """
    Compute the relative amount of time each instrument sounds relative to the others
    :param midi: to slice
    :param num_slices: to split the midi into
    :param midi_end_time: time where midi ends
    :param instrument_indexes: computed by the method "get_midis_and_texts"
    :return: array where each entry is a list of size num_instruments with relative instrument presence in each slice
    """
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
    """
    Get a matrix that indicates in each timestep which instruments are sounding
    :param midi: to slice
    :param midi_end_time: time where midi ends
    :param instrument_indexes: computed by the method "get_midis_and_texts"
    :return: matrix of dim [num_instruments, total midi time] with note markings for each instrument
    """
    note_matrix = np.zeros([len(instrument_indexes), int(midi_end_time)])
    for instrument in midi.instruments:
        row = note_matrix[instrument_indexes[instrument.program]]
        for note in instrument.notes:
            row[int(note.start): int(note.end) + 1] = 1
    return note_matrix


def get_set(texts, midi_indexes, slices_indexes, word_indexes, nlp):
    """
    Get a set ready to be fed into the RNN
    :param texts: lyrics of the songs in the set
    :param midi_indexes: indexes of midi embeddings in set
    :param slices_indexes: indexes of midi slice embeddings in set
    :param word_indexes: indexes of word in whole vocabulary
    :param nlp: the Spacy model
    :return: array with four elements, the indexes of [words, midis, midi slices, target classes] in the set
    """
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
    """
    Get the indexes of midi slices in set, to be used by the method "get_set"
    :param words:
    :param slice_indexes:
    :return:
    """
    x_midi_sliced = []
    midi_slice_idx = 0
    for word in words:
        x_midi_sliced.append(slice_indexes[midi_slice_idx])
        if word == '&':  # end of phrase
            midi_slice_idx += 1
    return x_midi_sliced


def build_model(word_embeddings, midi_embeddings, lstm_lens, dense_lens, mode, slice_embeddings):
    """
    Build the lyric generating model, employs the Adam optimizer and sparse categorical cross-entropy
    :param word_embeddings: np.array with the word embeddings
    :param midi_embeddings: np.array with the midi embeddings
    :param lstm_lens: array with the size of each lstm layer to process words and midi slices
    :param dense_lens: array with the size of each dense layer to process the concatenated vector
    :param mode: 'no midi' -> ignore midi and slices, 'simple' -> ignore slices, 'complex' -> ignore nothing
    :param slice_embeddings: np.array with the midi slice embeddings
    :return: compiled model
    """
    # word component
    word_in = Input(shape=(1,), name='word_in')
    lstm_word = Embedding(word_embeddings.shape[0], word_embeddings.shape[1],
                          trainable=False, weights=[word_embeddings], input_length=1)(word_in)
    # generate lstm layers in a loop
    for i, lstm_len in enumerate(lstm_lens):
        if i > 0:  # for getting the right dimensionality
            lstm_word = Reshape(tuple([1, lstm_word.shape[1]]))(lstm_word)
        lstm_word = LSTM(lstm_len, dropout=0.1)(lstm_word)

    if mode != 'no midi':
        # midi component
        midi_in = Input(shape=(1,), name='midi_in')
        midi_embedding = Embedding(midi_embeddings.shape[0], midi_embeddings.shape[1],
                            trainable=False, weights=[midi_embeddings], input_length=1)(midi_in)
        midi_embedding = Reshape(tuple([midi_embedding.shape[2]]))(midi_embedding)

        # concatenation
        if mode == 'complex':
            # midi slice component
            slice_in = Input(shape=(1,), name='slice_in')
            lstm_slice = Embedding(slice_embeddings.shape[0], slice_embeddings.shape[1],
                                  trainable=False, weights=[slice_embeddings], input_length=1)(slice_in)
            # generate lstm layers in a loop
            for i, lstm_len in enumerate(lstm_lens):
                if i > 0:  # for getting the right dimensionality
                    lstm_slice = Reshape(tuple([1, lstm_slice.shape[1]]))(lstm_slice)
                lstm_slice = LSTM(lstm_len, dropout=0.1)(lstm_slice)
            dense = concatenate([lstm_word, midi_embedding, lstm_slice])
        else:
            dense = concatenate([lstm_word, midi_embedding])
    else:  # no concatenation since midi and midi slices are ignored
        dense = lstm_word
    # generate dense layers in a loop
    for dense_len in dense_lens:
        dense = Dense(dense_len)(dense)
        dense = Dropout(0.1)(dense)
    output = Dense(word_embeddings.shape[0], activation='softmax')(dense)
    if mode == 'no midi':
        model_input = word_in
    elif mode == 'simple':
        model_input = [word_in, midi_in]
    else:
        model_input = [word_in, midi_in, slice_in]
    model = Model(model_input, output, name=mode)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    return model


def generate_texts(model, x_test, vocabulary, first_word_idx):
    """
    Generate lyrics that start with a particular word, for all instances in test
    :param model: lyric generating model
    :param x_test: input for the model that corresponds to the test set
    :param vocabulary: output from the method "get_vocabulary_and_word_embeddings"
    :param first_word_idx: index of the first word for the generated lyrics
    :return: list with generated lyrics for each song in test set
    """
    mode = model.name
    generated_texts = []
    prev_midi_idx = -1
    idx_range = range(len(vocabulary))
    for i in range(len(x_test[0])):
        curr_midi_idx = x_test[1][i]
        if mode == 'complex':
            midi_slice_idx = x_test[2][i]
        if prev_midi_idx != curr_midi_idx:  # got to next song
            prev_midi_idx = curr_midi_idx
            if i != 0:  # save text generated from last song
                text = ' '.join(generated_text)
                print(text)
                generated_texts.append(text)
            print('test song %d' % (len(generated_texts) + 1))
            word_idx = first_word_idx
            generated_text = [vocabulary[word_idx]]
            model.reset_states()
        if mode == 'no midi':
            pred_word_idxs = model.predict([[word_idx]])
        elif mode == 'simple':
            pred_word_idxs = model.predict([[word_idx], [curr_midi_idx]])
        else:
            pred_word_idxs = model.predict([[word_idx], [curr_midi_idx], [midi_slice_idx]])
        word_idx = int(choice(idx_range, 1, p=pred_word_idxs.reshape(-1)))
        generated_text.append(vocabulary[word_idx])
    text = ' '.join(generated_text)
    print(text)
    generated_texts.append(text)  # append last generated text
    return generated_texts


def write_to_file(history, generated_texts, layer_sizes, iteration, model_name):
    """
    Write the results to a file
    :param history: output from model.fit
    :param generated_texts: output from method "generate_texts"
    :param layer_sizes: to differentiate folder
    :param iteration: to differentiate folder
    :param model_name: to differentiate folder
    """
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

# todo: run "pip install pretty_midi" in terminal
# todo: run "python -m spacy download en_core_web_md" in terminal
# todo: select experiment parameters:
# base_dir = 'DL3/'  # for colab
base_dir = ''  # for pc
train_subset = 0  # for speeding up debugging. select 0 for whole train set
val_split = 0.2
# mode = 'no midi'
mode = 'simple'
# mode = 'complex'
lstm_lens = [128]
dense_lens = [512]
epochs = 30
batch_size = 256

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

train_len = len(texts_train)
midi_indexes_train = list(range(train_len))
midi_indexes_test = list(range(train_len, train_len + len(texts_test)))
val_len = int(len(texts_train) * val_split)
print('building train set...')
x_train, y_train = get_set(texts_train[:-val_len], midi_indexes_train[:-val_len], slice_indexes_train[:-val_len],
                           word_indexes, nlp)
print('building validation set...')
x_val, y_val = get_set(texts_train[-val_len:], midi_indexes_train[-val_len:], slice_indexes_train[-val_len:],
                       word_indexes, nlp)
print('building test set...')
x_test, y_test = get_set(texts_test, midi_indexes_test, slice_indexes_test, word_indexes, nlp)

# build model
model = build_model(word_embeddings, midi_embeddings, lstm_lens, dense_lens, mode, midi_sliced_embeddings)
model.summary()
checkpoint = ModelCheckpoint('checkpoint.h5', save_best_only=True)
if mode == 'no midi':
  x_train_mode, x_val_mode = x_train[0], x_val[0]
elif mode == 'simple':
  x_train_mode, x_val_mode = x_train[:-1], x_val[:-1]
else:
  x_train_mode, x_val_mode = x_train, x_val
history = model.fit(x_train_mode, y_train, batch_size, epochs, callbacks=[checkpoint], validation_data=(x_val_mode, y_val))
model = load_model('checkpoint.h5')

# generate text for all the 5 test melodies, 3 times
first_word_idxs = [word_indexes[word] for word in ['the', 'you', 'world']]
for i, first_word_idx in enumerate(first_word_idxs):
    print('iteration %d/3' % (i + 1))
    generated_texts = generate_texts(model, x_test, vocabulary, first_word_idx)
    write_to_file(history.history, generated_texts, lstm_lens, i + 1, model.name)

print('\ndone')

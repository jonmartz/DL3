import pretty_midi
import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
# %matplotlib inline
# For putting audio in the notebook
# import IPython.display
import spacy
import collections
import tensorflow as tf
from keras import layers


def build_model():
    model = tf.keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))

    model.summary()


def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


def pretty_midi_test():
    pm = pretty_midi.PrettyMIDI('dataset/midi_files/2_Unlimited_-_Get_Ready_for_This.mid')

    # plt.figure(figsize=(12, 4))
    # plot_piano_roll(pm, 24, 84)
    # plt.show()

    # Let's look at what's in this MIDI file
    print('There are {} time signature changes'.format(len(pm.time_signature_changes)))
    print('There are {} instruments'.format(len(pm.instruments)))
    print('Instrument 3 has {} notes'.format(len(pm.instruments[0].notes)))
    print('Instrument 4 has {} pitch bends'.format(len(pm.instruments[4].pitch_bends)))
    print('Instrument 5 has {} control changes'.format(len(pm.instruments[5].control_changes)))


def spacy_test():

    # Load English tokenizer, tagger, parser, NER and word vectors
    nlp = spacy.load('en_core_web_md')

    # Process whole documents
    text = ("When Sebastian Thrun started working on self-driving cars at "
            "Google in 2007, few people outside of the company took him "
            "seriously. “I can tell you very senior CEOs of major American "
            "car companies would shake my hand and turn away because I wasn’t "
            "worth talking to,” said Thrun, in an interview with Recode earlier "
            "this week.")

    # process a sentence using the model
    doc = nlp(text)

    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

    # Find named entities, phrases and concepts
    for entity in doc.ents:
        print(entity.text, entity.label_)

    # Getting neighborhood:
    dog_cluster = nlp.vocab['dog'].cluster

    # Get the mean vector for the entire sentence (useful for sentence classification etc.)
    # doc.vector
    print('done')


spacy_test()

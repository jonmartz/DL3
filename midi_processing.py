import numpy as np
import pandas as pd
import pretty_midi
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
from mido.midifiles.meta import KeySignatureError


def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


def pretty_midi_test():
    pm = pretty_midi.PrettyMIDI('dataset/midi_files/2_Unlimited_-_Get_Ready_for_This.mid')

    # plot piano roll
    plt.figure(figsize=(12, 4))
    plot_piano_roll(pm, 24, 84)
    plt.show()

    # Let's look at what's in this MIDI file
    print('There are {} time signature changes'.format(len(pm.time_signature_changes)))
    print('There are {} instruments'.format(len(pm.instruments)))
    print('Instrument 3 has {} notes'.format(len(pm.instruments[0].notes)))  # todo: use this feature, weighted presence of each instrument
    print('Instrument 4 has {} pitch bends'.format(len(pm.instruments[4].pitch_bends)))
    print('Instrument 5 has {} control changes'.format(len(pm.instruments[5].control_changes)))

    # plot tempo changes
    tempo_change_times, tempos = pm.get_tempo_changes()  # todo: use this to get tempo by time
    plt.plot(tempo_change_times, tempos, '.')
    plt.xlabel('Time')
    plt.ylabel('Tempo')
    plt.show()

    # Get and downbeat times
    beats = pm.get_beats()
    downbeats = pm.get_downbeats()
    # Plot piano roll
    plt.figure(figsize=(12, 4))
    plot_piano_roll(pm, 24, 84)
    ymin, ymax = plt.ylim()
    # Plot beats as grey lines, downbeats as white lines
    mir_eval.display.events(beats, base=ymin, height=ymax, color='#AAAAAA')
    mir_eval.display.events(downbeats, base=ymin, height=ymax, color='#FFFFFF', lw=2)
    # Only display 20 seconds for clarity
    plt.xlim(25, 45)
    plt.show()

    # todo: use this feature, can slice chroma to get song subset and also do per instrument
    # a proxy for key
    total_velocity = sum(sum(pm.get_chroma()))
    print([sum(semitone) / total_velocity for semitone in pm.get_chroma()])

    # Plot a pitch class distribution - sort of a proxy for key
    plt.bar(np.arange(12), pm.get_pitch_class_histogram())
    plt.xticks(np.arange(12), ['C', '', 'D', '', 'E', 'F', '', 'G', '', 'A', '', 'B'])
    plt.xlabel('Note')
    plt.ylabel('Proportion')
    plt.show()
    # Let's count the number of transitions from C to D in this song
    n_c_to_d = 0
    for instrument in pm.instruments:
        # Drum instrument notes don't have pitches!
        if instrument.is_drum:
            continue
        for first_note, second_note in zip(instrument.notes[:-1], instrument.notes[1:]):
            n_c_to_d += (first_note.pitch % 12 == 0) and (second_note.pitch % 12 == 2)
    print('{} C-to-D transitions.'.format(n_c_to_d))

    # general analysis
    print('an empirical estimate of its global tempo:')
    print(pm.estimate_tempo())  # todo: use this feature
    print('Compute the relative amount of each semitone:')
    chroma = pm.get_chroma()
    total_velocity = sum(sum(chroma))
    print([sum(semitone) / total_velocity for semitone in chroma])


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


# pretty_midi_test()

# todo: features to use:
""" 
tempo of song (pm.estimate_tempo())
vector of instrument weighted presence (len(pm.instruments[0].notes) / sum(len(pm.instruments[i].notes) for i in pm.instruments))
vector of average semi-key for each instrument (get_pitch_class_histogram(use_duration=True, use_velocity=True, normalize=True))
"""

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
print('done')


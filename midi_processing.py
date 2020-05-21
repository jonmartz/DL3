import numpy as np
import pretty_midi
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt


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
    print('There are {} time signature changes'.format(len(pm.time_signature_changes)))  # todo: use this feature
    print('There are {} instruments'.format(len(pm.instruments)))
    print('Instrument 3 has {} notes'.format(len(pm.instruments[0].notes)))  # todo: use this feature, weighted presence of each instrument
    print('Instrument 4 has {} pitch bends'.format(len(pm.instruments[4].pitch_bends)))
    print('Instrument 5 has {} control changes'.format(len(pm.instruments[5].control_changes)))

    # plot tempo changes
    times, tempo_changes = pm.get_tempo_changes()
    plt.plot(times, tempo_changes, '.')
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

    # Plot a pitch class distribution - sort of a proxy for key
    plt.bar(np.arange(12), pm.get_pitch_class_histogram())  # todo: use this feature
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


pretty_midi_test()

# todo: features to use, beside the ones above
""" 
vector of instruments present
for instruments: get_pitch_class_histogram(use_duration=False, use_velocity=False, normalize=False)
"""


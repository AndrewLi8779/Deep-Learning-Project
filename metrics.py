import sys
import mir_eval
import numpy as np

sys.path.append('C:\\Users\\Andrew\\Documents\\GitHub\\Deep-Learning-Project\\preprocessing')

from preprocessing.audio_preprocessor import load_metadata
from preprocessing.midi_preprocessor import decode_midi, Event, PAD, SOS, EOS, event_seq_to_snote_seq, encode_midi


def _make_evaluation_arrays(idx_array, label):
    est_midi = decode_midi(idx_array)
    ref_midi = decode_midi(label)

    ref_intervals = []
    ref_pitches = []
    ref_velocity = []
    for note in ref_midi:
        ref_intervals.append([note.start, note.end])
        ref_pitches.append(note.pitch)
        ref_velocity.append(note.velocity)

    est_intervals = []
    est_pitches = []
    est_velocity = []
    for note in est_midi:
        est_intervals.append([note.start, note.end])
        est_pitches.append(note.pitch)
        est_velocity.append(note.velocity)

    ref_intervals = np.array(ref_intervals)
    ref_pitches = np.array(ref_pitches)
    ref_velocity = np.array(ref_velocity)
    est_intervals = np.array(est_intervals)
    est_pitches = np.array(est_pitches)
    est_velocity = np.array(est_velocity)

    return ref_intervals, ref_pitches, ref_velocity, est_intervals, est_pitches, est_velocity


def note_on_f1(ref_intervals, ref_pitches, est_intervals, est_pitches):
    return mir_eval.transcription.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None)


def note_on_off_f1(ref_intervals, ref_pitches, est_intervals, est_pitches):
    return mir_eval.transcription.precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches)


def note_on_off_velocity_f1(ref_intervals, ref_pitches, ref_velocity, est_intervals, est_pitches, est_velocity):
    return mir_eval.transcription_velocity.precision_recall_f1_overlap(ref_intervals, ref_pitches, ref_velocity, est_intervals, est_pitches, est_velocity)


def all_in_one_f1(idx_array, label):
    ref_intervals, ref_pitches, ref_velocity, est_intervals, est_pitches, est_velocity = _make_evaluation_arrays(idx_array, label)
    note_on = note_on_f1(ref_intervals, ref_pitches, est_intervals, est_pitches)
    note_on_off = note_on_off_f1(ref_intervals, ref_pitches, est_intervals, est_pitches)
    note_on_off_velocity = note_on_off_velocity_f1(ref_intervals, ref_pitches, ref_velocity, est_intervals, est_pitches, est_velocity)

    return note_on, note_on_off, note_on_off_velocity


if __name__ == '__main__':
    data_path = './data/maestro-v3.0.0/'
    meta = load_metadata(data_path + 'maestro-v3.0.0.json')
    #
    # encoded = encode_midi(data_path + meta['midi_filename']['158'])
    # encoded2 = encode_midi(data_path + meta['midi_filename']['157'])
    #
    # print(all_in_one_f1(encoded, encoded2))

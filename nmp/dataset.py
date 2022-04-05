#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset functions."""
import pandas as pd
import numpy as np
import pypianoroll
import random
import copy
import tensorflow as tf


def transpose(data):
    """Transpose time and pitch."""
    df = pd.DataFrame(data)
    return df.T.values


def binarize(data):
    """Binarize data."""
    return np.array([[1 if e else 0 for e in r] for r in data])


def threshold(data, thresh=0.5):
    """Convert data from sigmoid output to 0-1."""
    return np.array([[1 if e > thresh else 0 for e in r] for r in data])


def ranked_threshold(data, steps, how_many=3):
    """Convert data from sigmoid output to 0-1 using most likely notes."""
    pr = copy.deepcopy(data)
    n = int(pr.shape[1]/steps)
    for t in range(pr.shape[0]):
        for step in range(steps):
            array = pr[t, step*n:(step+1)*n]
            vect = copy.deepcopy(array)  # Deep copy
            vect.sort()  # Can be done thanks to deep copy
            thresh = vect[-how_many-1]  # Last values
            pr[t, step*n:(step+1)*n] = np.array([1 if e > thresh
                                                 else 0 for e in array])

    return pr


def pad_piano_roll(pr, low_lim=21, high_lim=109):
    """Convert 88-notes piano roll to 128-notes piano roll.

    Useful for plotting using pypianoroll library.
    """
    L = len(pr)
    pad1 = np.zeros((L, low_lim))
    pad2 = np.zeros((L, 128 - high_lim))
    pr = np.concatenate((pad1, pr, pad2), axis=1)
    return pr


def pyplot_piano_roll(pr, cmap="Blues", br=None, db=None, low_lim=21,
                      high_lim=109, ax=None):
    """Plot piano roll representation."""
    pr = pad_piano_roll(pr, low_lim, high_lim)
    pr = pypianoroll.Track(pianoroll=pr)
    fig, ax = pypianoroll.plot_track(pr, cmap=cmap,
                                     beat_resolution=br, downbeats=db)
    return fig, ax


def import_one(filename, beat_resolution, binarize=0):
    """Import one file and generate a piano roll."""
    pr = pypianoroll.parse(filename, beat_resolution)

    if binarize:
        pr = pypianoroll.binarize(pr)

    merged = pr.get_merged_pianoroll()

    return merged


def import_one_choral(song):
    """Convert choral into a piano_roll."""
    pr = np.zeros((len(song), 128))
    for c, t in enumerate(pr):
        t[list(map(int, song[c]))] = 1
    return pr


def write_midi(data, filename, low_lim, high_lim, tempo=120.0, br=2):
    """Save piano roll to a midi file."""
    pr = copy.deepcopy(data)
    pr[pr > 0] = 127
    track = pypianoroll.Track(pad_piano_roll(pr, low_lim, high_lim))
    multitrack = pypianoroll.Multitrack(tracks=[track], tempo=tempo,
                                        beat_resolution=br)
    multitrack.write(filename)


class Dataset:
    """Class to yield midi_file as batches."""

    def __init__(self, midi_list, path, fs=4, quant=0, binarize=1, bl=0):
        """Initialise list."""
        self.midi_list = midi_list
        self.dim = len(midi_list)
        self.fs = fs
        self.quant = quant
        self.binar = binarize
        self.path = path
        self.baseline = bl
        self.data = []
        self.targets = []

    def binarize(self, pr):
        """Binarize piano roll matrix and return it."""
        return np.array([[np.float32(1) if e else np.float32(0)
                         for e in r] for r in pr])

    def quantize(self, obj):
        """Quantize time axis."""
        PPQ = obj.resolution
        for instr in obj.instruments:
            for note in instr.notes:
                note.start = obj.time_to_tick(note.start)/PPQ
                note.end = obj.time_to_tick(note.end)/PPQ

        return obj

    # def pitch_transpose(self, ds, value, timesteps=None):
    #     """Perform pitch transposition."""
    #     if timesteps:
    #         range_timesteps = ds.shape[1]
    #         for t in range(range_timesteps):
    #             d = pd.DataFrame(ds[:, t, :])
    #             d.shift(value, axis=1, fill_value=0)
    #             ds[:, t, :] = d.values
    #     else:
    #         d = pd.DataFrame(ds)
    #         d.shift(value, axis=1, fill_value=0)
    #         ds = d.values
    #
    #     return ds

    # def generate(self, bs, limit=None, trans=None, name=''):
    #     """Yield a dataset."""
    #     dim = self.dataset[0].shape[0]
    #     batch = bs
    #     epoch = 0
    #     while True:
    #         cnt = 0
    #         epoch += 1
    #         en = 0
    #         c = 0
    #         while en < dim:
    #             en = c + batch
    #             yield (self.dataset[0][c:en], self.dataset[1][c:en])
    #             c += batch
    #             cnt += 1
    #         if trans == 1:
    #             for v in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
    #                 x = self.pitch_transpose(self.dataset[0], v, 1)
    #                 y = self.pitch_transpose(self.dataset[1], v)
    #                 en = 0
    #                 c = 0
    #                 while en < dim:
    #                     en = c + batch
    #                     yield (x[c:en], y[c:en])
    #                     c += batch
    #                     cnt += 1
    #         print("\nEpoch %d finished for %s, %d" % (epoch, name, cnt))
    #         # if limit == 1:
    #         #     break

    def build_dataset(self, name, step, t_step, steps, down, transpose=0,
                      low_lim=21, high_lim=109):
        """Build a dataset."""
        print("Building %s dataset (%d files)" % (name, len(self.midi_list)))
        for m in self.midi_list:
            prt = import_one(str(self.path / m), beat_resolution=self.fs,
                             binarize=0)
            prt = prt[:, low_lim:high_lim]  # Crop piano roll
            prt = self.binarize(prt)

            if down > 1:
                prt = downsample_roll(prt, steps, down)

            if self.baseline:
                data = prt[:-t_step, :]
                target = prt[step-1:-1, :]

            else:
                data = prt[:-t_step, :]
                target = prt[step:, :]

            if step > 0:
                data_new = []
                for c in range(len(data)-step+1):
                    conc = np.array([data[x] for x in range(c, c+step)])
#                     conc = downsample_one(conc, steps, down)
                    data_new.append(conc)

                data = np.array(data_new)

            if t_step > 0:
                target_new = []

                if self.baseline:
                    for c in range(len(target)-t_step+1):
                        conc = np.array([target[c] for _ in range(c,
                                                                  c+t_step)])

#                         conc = downsample_one(conc, steps, down)
                        target_new.append(np.concatenate(conc, axis=None))

                else:
                    for c in range(len(target)-t_step+1):
                        conc = np.array([target[x] for x in range(c,
                                                                  c+t_step)])

#                         conc = downsample_one(conc, steps, down)
                        target_new.append(np.concatenate(conc, axis=None))

                target = np.array(target_new)

            self.data.append(data)
            self.targets.append(target)

        self.concatenate_all()

    def build_rnn_dataset(self, name, down, low_lim=21, high_lim=109):
        """Build a dataset."""
        print("Building %s dataset (%d files)" % (name, len(self.midi_list)))

        for m in self.midi_list:
            prt = import_one(str(self.path / m), beat_resolution=self.fs,
                             binarize=0)
            prt = prt[:, low_lim:high_lim]  # Crop piano roll
            prt = self.binarize(prt)

            if down > 1:
                prt = downsample_roll(prt, 1, down)

            if self.baseline:
                data = prt[:-1, :]
                # target = []
                # target = prt[:-1, :]

            # Legacy.
            # else:
            #     data = prt[:-1, :]
            #     target = []
            #     # target = prt[1:, :]

            # 10 to predict 10.
            else:
                data = prt
                # target = []

            data_new = []
            for c in range(len(data) - 19):
                conc = np.array([data[x] for x in range(c, c+20)])
                conc = np.concatenate([x for x in conc], axis=0)
                data_new.append(conc)

            data = np.array(data_new)

            self.data.append(data)
            self.targets.append(data)
            # self.targets.append(target)

        self.concatenate_all()

    def concatenate_all(self):
        """Build dataset by concatenating all files."""
        self.data2 = np.concatenate([x for x in self.data], axis=0)
        self.targets2 = np.concatenate([x for x in self.targets], axis=0)
        self.dataset = (self.data2, self.targets2)
        del self.data2
        del self.targets2


def random_baseline(length, num_notes, select=None):
    """Generate random baseline"""
    base = np.zeros((length, num_notes))

    if select is None:
        select = list(range(num_notes))

    else:
        select = list(set(select))

    for t in base:
        for _ in range(3):
            choice = random.choice(select)
            t[choice] = 1

    return pd.DataFrame(base)


def hold_baseline(length, num_notes, select):
    """Generate hold baseline"""
    base = np.zeros((length, num_notes))

    select = list(set(select))

    for t in base:
        for choice in select:
            t[choice] = 1

    return pd.DataFrame(base)


def get_indexes(pr):
    """Return indexes from piano roll structure.

    select: all notes played in the piano roll for every timestep
    noteset: set of notes played in the piano roll
    """
    pr = pd.DataFrame(pr)
    serie = pr.apply(lambda row: row[row == 1].index, axis=1)
    select = []
    for x in serie:
        select.append([x[y] for y in range(len(x))])

    noteset = []
    for x in select:
        noteset += [x[y] for y in range(len(x))]
    noteset = list(set(noteset))

    return (select, noteset)


def generate(datasets, bs=64, trans=0):
    """Yield dataset with random order and transposition."""
    length = datasets[0].shape[0]
    randomize = list(range(length))
    transpositions = list(range(-5, 7))

    while True:
        random.shuffle(randomize)
        b = 0

        while True:
            batch = randomize[b:b+bs]
            b += bs

            if datasets[0][batch, :, :].shape[0] == 0:
                break

            if trans:
                trans = random.choice(transpositions)
                for i in range(datasets[0][batch, :, :].shape[1]):
                    datasets[0][batch, i, :] = pitch_trans(datasets[0][batch,
                                                                       i, :],
                                                           trans)
                datasets[1][batch, :] = pitch_trans(datasets[1][batch, :],
                                                    trans)

            x = datasets[0][batch, :, :]
            y = datasets[1][batch, :]

            yield (x, y, [None])


def gener(data):
    for x in data:
        yield x


def generate_slices(datasets, bs=64, trans=0):
    """Yield dataset with random order and transposition."""
    seq_length = 64
    # data = [(x, y) for x, y in zip(datasets[0], datasets[1])]
    data = datasets[0]
    data = tf.data.Dataset.from_tensor_slices(data)
    x, y = slice_dataset(datasets)
    x = x.batch(seq_length, drop_remainder=True)
    y = y.batch(seq_length, drop_remainder=True)
    gen = gener(list(y.as_numpy_iterator()))
    dataset = x.map(lambda x: (x, next(gen)))
    return dataset

    # length = datasets[0].shape[0]
    # randomize = list(range(length))
    # transpositions = list(range(-5, 7))

    # while True:
    #     random.shuffle(randomize)
    #     b = 0

    #     while True:
    #         batch = randomize[b:b+bs]
    #         b += bs

    #         if datasets[0][batch, :, :].shape[0] == 0:
    #             break

    #         if trans:
    #             trans = random.choice(transpositions)
    #             for i in range(datasets[0][batch, :, :].shape[1]):
    #                 datasets[0][batch, i, :] = pitch_trans(datasets[0][batch,
    #                                                                    i, :],
    #                                                        trans)
    #             datasets[1][batch, :] = pitch_trans(datasets[1][batch, :],
    #                                                 trans)

    #         x = datasets[0][batch, :, :]
    #         y = datasets[1][batch, :]

    #         yield (x, y, [None])


def generate_on_batch(datasets, bs=64, trans=0):
    """Yield dataset with random order and transposition."""
    transpositions = list(range(-5, 7))

    cnt = 0
    while cnt == 0:
        s = 0
        b = 0
        for x, y in zip(datasets[0], datasets[1]):
            s += 1
            # print("Song n. %d" % s)
            next_ = 0
            while True:
                if next_ == 1:
                    next_ = 0
                    break

                length = x.shape[0]
                # print(length)
                randomize = list(range(length))
                randomize = [r*10 for r in randomize]
                batch = randomize[b:b+bs]
                # print("Using batch:", randomize[b:b+bs])
                b += bs

                if trans:
                    trans = random.choice(transpositions)

                    try:
                        for i in range(x[batch, :, :].shape[1]):
                            x[batch, i, :] = pitch_trans(x[batch, i, :], trans)

                        y[batch, :] = pitch_trans(y[batch, :], trans)

                    except IndexError:
                        b = 0
                        next_ = 1
                        yield (np.array([[[-1]]]), np.array([[[-1]]]))

                # if x[batch, :, :].shape[0] == bs:
                #     print(x[batch, :, :].shape[0])
                #     yield (x[batch, :, :], y[batch, :])

                # else:
                #     b = 0
                #     next_ = 1
                #     yield (np.array([[[-1]]]), np.array([[[-1]]]))

                try:
                    x[batch, :, :].shape[0] == bs
                    # print(x[batch, :, :].shape[0])
                    yield (x[batch, :, :], y[batch, :])

                except IndexError:
                    b = 0
                    next_ = 1
                    yield (np.array([[[-1]]]), np.array([[[-1]]]))

        cnt = 1
        print("Dataset finished")


def generate_stateful(datasets, bs=64, trans=0):
    """Yield dataset with random order and transposition."""
    length = datasets[0].shape[0]
    randomize = list(range(length))
    randomize = [r*1 for r in randomize]
    transpositions = list(range(-5, 7))
    trans = random.choice(transpositions)
    b = 0
    random.shuffle(randomize)
    while True:
        batch = randomize[b:b+bs]
        b += bs

        try:
            shap = datasets[0][batch, :, :].shape[0]

        except Exception:
            shap = -1

        if shap != bs:
            break

        if trans:
            for i in range(datasets[0][batch, :, :].shape[1]):
                datasets[0][batch, i, :] = pitch_trans(datasets[0][batch,
                                                                   i, :],
                                                       trans)
            datasets[1][batch, :] = pitch_trans(datasets[1][batch, :],
                                                trans)

        x = datasets[0][batch, :, :]
        y = datasets[1][batch, :]

        yield (x, y, [None])


def pitch_trans(data, value):
    """Perform music transposition."""
    data = pd.DataFrame(data)
    data = data.shift(value, axis=1, fill_value=0)
    return np.array(data.values)


def downsample_roll(pr, steps, down):
    """Downsample piano roll.

    This is done in order to facilitate predictions.
    """
    if down > 1:
        cnt = 0
        array = []
        for i in range(int(len(pr)/down)):
            array.append(pr[cnt:cnt+down, :].any(axis=0))
            cnt += down
        if len(pr[cnt:]) > 0:
            array.append(pr[cnt:].any(axis=0))  # Last

        return np.array(array)


def upsample_roll(pr, steps, up):
    """Upsample prediction piano roll.

    This is done in order to fit the original piano roll of the song since it
    was downsampled to make predictions.
    """
    array = []
    for t in pr:
        for _ in range(up):
            array.append(t)

    return np.array(array)


def slice_dataset(datasets):
    """Slice dataset with tensorflow."""
    x = datasets[0]
    y = datasets[1]
    x = tf.data.Dataset.from_tensor_slices(x)
    y = tf.data.Dataset.from_tensor_slices(y)
    return (x, y)


def fill_gap(pr, model, position, size=10, num_notes=64, baseline=0,
             how_many=2):
    """Fill holes in a song with predictions.

    Feedforward model is used.
    """
    start = position - 120
    end = position

    past = downsample_roll(pr[start:end, :], 10, 12)
    past = np.array([past])

    if baseline:
        _, noteset = get_indexes(past[0])
        base = random_baseline(10, num_notes, noteset)
        base = np.array(base)
        upsampled = upsample_roll(base, 10, 12)

    else:
        predictions = model.predict(past)
        predictions_bin = ranked_threshold(predictions, steps=10,
                                           how_many=how_many)
        predictions_bin = predictions_bin.reshape((10, 64))
        upsampled = copy.deepcopy(predictions_bin)
        upsampled = upsample_roll(upsampled, 10, 12)

    filled = copy.deepcopy(pr)
    filled[end: end+12*size] = upsampled[:12*size]

    return filled


def split_input_target(chunk):
    """Map instances into a tuple (input, target)."""
    input_notes = chunk[:-1]
    target_notes = chunk[1:]
    return input_notes, target_notes


def fill_gap_rnn(pr, model, position, size=1, num_notes=64, baseline=0,
                 how_many=2):
    """Fill holes in a song with predictions.

    Recurrent model is used.
    """
    start = position - 120
    end = position

    past = downsample_roll(pr[start:end, :], 0, 12)
    generated = []

    input_batch = tf.expand_dims(past, 0)
    model.reset_states()

    for i in range(size):
        predictions = model(tf.cast(input_batch, tf.float32))
        pred = np.array(tf.squeeze(predictions, 0))
        predictions_bin = ranked_threshold(pred, steps=1, how_many=how_many)

        generated.append(np.array(predictions_bin[-1]))
        input_batch = tf.expand_dims([predictions_bin[-1]], 0)

    upsampled = copy.deepcopy(generated)
    upsampled = upsample_roll(upsampled, 10, 12)

    filled = copy.deepcopy(pr)
    filled[end: end+12*size] = upsampled

    return filled


def fill_gap_hold(pr, position, size=10, num_notes=64):
    """Fill holes in a song with predictions.

    Feedforward model is used.
    """
    start = position - 120
    end = position

    past = downsample_roll(pr[start:end, :], 10, 12)
    past = np.array([past])
    select, noteset = get_indexes(past[0])
    base = hold_baseline(10, num_notes, select[-1])
    base = np.array(base)
    upsampled = upsample_roll(base, 10, 12)

    filled = copy.deepcopy(pr)
    filled[end: end+12*size] = upsampled[:12*size]

    return filled


def predict_rnn2(pr, model, size=10, num_notes=64, how_many=5):
    """Fill holes in a song with predictions.

    Recurrent model is used.
    Predictions are re-used for next predictions (biased).
    """
    generated = []
    generated_cont = []

    input_batch = tf.expand_dims(pr, 0)
    model.reset_states()

    for i in range(size):
        predictions = model(tf.cast(input_batch, tf.float32))
        pred = np.array(tf.squeeze(predictions, 0))

        generated_cont.append(np.array(pred[-1]))
        predictions_bin = ranked_threshold(pred, steps=1, how_many=how_many)

        generated.append(np.array(predictions_bin[-1]))
        input_batch = tf.expand_dims([predictions_bin[-1]], 0)

    # upsampled = copy.deepcopy(generated)
    # upsampled = upsample_roll(upsampled, 10, 12)

    # filled = copy.deepcopy(pr)
    # filled[end: end+12*size] = upsampled

    return np.array(generated_cont)


def predict_rnn(pr, model, size=10, num_notes=64, how_many=5):
    """Fill holes in a song with predictions.

    Recurrent model is used.
    """
    generated = []
    generated_cont = []

    input_batch = tf.expand_dims(pr, 0)
    model.reset_states()

    predictions = model(tf.cast(input_batch, tf.float32))
    pred = np.array(tf.squeeze(predictions, 0))

    generated_cont.append(np.array(pred[-1]))
    predictions_bin = ranked_threshold(pred, steps=1, how_many=how_many)

    generated.append(np.array(predictions_bin[-1]))
    # input_batch = tf.expand_dims([predictions_bin[-1]], 0)

    # upsampled = copy.deepcopy(generated)
    # upsampled = upsample_roll(upsampled, 10, 12)

    # filled = copy.deepcopy(pr)
    # filled[end: end+12*size] = upsampled

    return np.array(generated_cont)


def fill_gap_rnn_final(pr, model, position, size=1, baseline=0,
                       how_many=2):
    """Fill holes in a song with predictions.

    Recurrent model is used. Long vector of notes predicted at once.
    """
    start = position - 120
    end = position

    past = downsample_roll(pr[start:end, :], 0, 12)
    generated = []
    past = np.reshape(past, (1, 640))
    input_batch = tf.expand_dims(past, 0)
    model.reset_states()
    predictions = model(tf.cast(input_batch, tf.float32))
    pred = np.array(tf.squeeze(predictions, 0))
    predictions_bin = ranked_threshold(pred, steps=size, how_many=how_many)
    generated.append(np.array(predictions_bin[-1]))
    generated = np.reshape(generated, (10, 64))
    upsampled = copy.deepcopy(generated)
    upsampled = upsample_roll(upsampled, 10, 12)
    filled = copy.deepcopy(pr)
    filled[end: end+12*size] = upsampled

    return filled

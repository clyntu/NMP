#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Display some predictions."""

import os
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
import tensorflow as tf
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from nmp import dataset
import copy
from nmp.dataset import downsample_roll, upsample_roll, ranked_threshold, pad_piano_roll, write_midi
from nmp import model as mod
import pypianoroll

# TensorFlow stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

BS = 64
FS = 24  # Sampling frequency. 10 Hz = 100 ms
Q = 0  # Quantize?
st = 10  # Past timesteps
num_ts = 10  # Predicted timesteps
DOWN = 12  # Downsampling factor
D = "data/POP909" # Dataset

# MODEL = 'lstm-501.h5'
MODEL = 'lstm-z-de.h5'

# 64 NOTES
LOW_LIM = 33  # A1
HIGH_LIM = 97  # C7

# LOW_LIM = 36  # A1
# HIGH_LIM = 85  # C7

NUM_NOTES = HIGH_LIM - LOW_LIM
CROP = [LOW_LIM, HIGH_LIM]  # Crop plots


P = Path(os.path.abspath(''))  # Compatible with Jupyter Notebook

# LOAD MODEL
model = load_model(filepath="models/"+MODEL, custom_objects=None,compile=True)

# checkpoint_dir = P / ('models/training_checkpoints/20220610-184001')
# model = mod.build_lstm_model(NUM_NOTES, 1)
# mod.compile_model(model, 'binary_crossentropy', 'adam',
#                   metrics=['accuracy'])
# model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# model.build(tf.TensorShape([1, None]))

# model.summary()

BASE = 0
hm = 4

# Get midi file 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="input Midi file for predictions")
args = parser.parse_args()
FILE = args.input_file

test_file = np.loadtxt(FILE)
# test_file = np.load(FILE)
# print("test_file shape: ", test_file.shape)

size = 1
steps = 1
how_many = 2

generated = []
generated_cont = []

# end = test_file.shape[0] - 120
# start = end - 120

end = 120
start = 0

# past = downsample_roll(test_file, 0, 12)
if test_file.shape[1] != 64:
    past = downsample_roll(test_file[start:end, LOW_LIM:HIGH_LIM], 0, 12)
    # print("downsampled test_file shape: ", past.shape)
else:
    past = downsample_roll(test_file[start:end, :], 0, 12)
    # print("downsampled test_file shape: ", past.shape) 
past = np.reshape(past, (1,640))
# print("past shape: ", past.shape)
input_batch = tf.expand_dims(past, 0)
model.reset_states()
predictions = model(tf.cast(input_batch, tf.float32))
pred = np.array(tf.squeeze(predictions, 0))

generated_cont.append(np.array(pred[-1]))
predictions_bin = ranked_threshold(pred, steps=steps, how_many=how_many)

generated.append(np.array(predictions_bin[-1]))

# size = 10
# predicted = dataset.predict_rnn(unpredicted, model, size=size, num_notes=NUM_NOTES, how_many=5)
# print("predicted_file shape: ", predicted.shape)

generated = np.reshape(generated, (10, 64))
upsampled = copy.deepcopy(generated)
upsampled = upsample_roll(upsampled, 10, 12)

# print("generated shape:", generated.shape)
# print("upsampled.shape:",upsampled.shape)

if test_file.shape[1] == 128:
    upsampled = pad_piano_roll(upsampled, LOW_LIM, HIGH_LIM)
    # print("padded upsampled.shape:",upsampled.shape)

filled = copy.deepcopy(test_file)
filled[end: end+120] = upsampled


if test_file.shape[1] == 64: 
    filled = pad_piano_roll(filled, LOW_LIM, HIGH_LIM)
# print("filled shape:", filled.shape)

f0 = copy.deepcopy(filled)
np.savetxt("out_"+FILE,f0)

tempo = 95.000143
write_midi(f0,"audio_output/out_"+FILE, 0, 128, tempo = tempo, br=24)
# # print(FILE)


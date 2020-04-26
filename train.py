#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main script."""
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import matplotlib.pyplot as plt
import os
import model as mod
import dataset
from dataset import plot_piano_roll

FS = 4  # Sampling frequency


def main():
    """Run main script."""
    # Build Keras model.
    model = mod.build_model()
    now = datetime.now()
    logger = TensorBoard(log_dir='logs/' + now.strftime("%Y%m%d-%H%M%S") +
                         "/", write_graph=True, update_freq='epoch')

    # Load midi files.
    midi_list = [x for x in os.listdir("./data/") if x.endswith('.mid')]
    epochs = 5
    train_list = midi_list[0:25]
    validation_list = midi_list[25:35]
    test_list = midi_list[50:51]
    print("Train list:  ", train_list)
    print("Validation list:  ", validation_list)
    print("Test list:  ", test_list)

    # Create generators.
    train = dataset.DataGenerator(train_list, fs=FS)
    validation = dataset.DataGenerator(validation_list, fs=FS)
    test = dataset.DataGenerator(test_list, fs=FS)

    # Fit the model.
    model.fit(train.generate(), epochs=epochs, steps_per_epoch=train.dim,
              validation_data=validation.generate(),
              validation_steps=validation.dim,
              callbacks=[logger])

    # Evaluate the model.
    print("Evaluation on test set:")
    _, prec, rec, f1 = model.evaluate(test.generate(), steps=test.dim)

    # Make predictions.
    predictions = model.predict(test.generate(), steps=test.dim)
    for c, t in enumerate(list(test.generate(limit=1))):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        plot_piano_roll(dataset.transpose(t[0]), 0, 128, ax1, FS)
        ax1.set_title('Test ' + str(c))

        plot_piano_roll(dataset.transpose(predictions), 0, 128, ax2, FS)
        ax2.set_title('Predictions ' + str(c))
        plt.show()


if __name__ == '__main__':
    main()
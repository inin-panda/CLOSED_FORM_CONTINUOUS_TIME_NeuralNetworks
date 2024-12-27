# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import argparse
from tf_cfc import CfcCell, MixedCfcCell, LTCCell
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

import sys

vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review


def load_imdb():
    """Load and preprocess IMDB dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=vocab_size
    )
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    
    # Split validation from training
    x_train, x_val = x_train[:-5000], x_train[-5000:]
    y_train, y_val = y_train[:-5000], y_train[-5000:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)



def eval(config, index_arg, verbose=0):
    (train_x, train_y), (test_x, test_y), (x_val, y_val) = load_imdb()
    
    if config.get("use_ltc"):
        cell = LTCCell(units=config["size"])
    elif config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)
    else:
        cell = CfcCell(units=config["size"], hparams=config)

    # pixel_input = tf.keras.Input(shape=(28 * 28, 1), name="pixel")
    inputs = tf.keras.layers.Input(shape=(maxlen,))
    token_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=config["embed_dim"]
    )
    cell_input = token_emb(inputs)
    cell_input = tf.keras.layers.Dropout(config["embed_dr"])(cell_input)

    rnn = tf.keras.layers.RNN(cell, return_sequences=False)
    dense_layer = tf.keras.layers.Dense(10)

    output_states = rnn(cell_input)
    y = dense_layer(output_states)

    model = tf.keras.Model(inputs, y)

    base_lr = config["base_lr"]
    decay_lr = config["decay_lr"]
    # end_lr = config["end_lr"]
    train_steps = train_x.shape[0] // config["batch_size"]
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )
    opt = (
        tf.keras.optimizers.Adam
        if config["optimizer"] == "adam"
        else tf.keras.optimizers.RMSprop
    )
    optimizer = opt(learning_rate_fn, clipnorm=config["clipnorm"])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Fit and evaluate
    hist = model.fit(
        x=train_x,
        y=train_y,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=(x_val, y_val),
        verbose=2,
    )
    
    
     # Visualize training history
    plot_history(hist.history)
    # test_accuracies = hist.history["val_sparse_categorical_accuracy"]
    # return np.max(test_accuracies)
    _, test_accuracy = model.evaluate(test_x, test_y, verbose=0)
    return hist.history, test_accuracy



BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "rmsprop",
    "batch_size": 32,
    "size": 64,
    "embed_dim": 32,
    "embed_dr": 0.3,
    "epochs": 20,
    "base_lr": 0.0005,
    "decay_lr": 0.8,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 1,
    "weight_decay": 0.00029,
    "use_mixed": True,
}

# 87.04% (MAX)
#  85.91% $\pm$ 0.99
BEST_DEFAULT = {
    "clipnorm": 10,
    "optimizer": "rmsprop",
    "batch_size": 32,
    "size": 192,
    "embed_dim": 192,
    "embed_dr": 0.0,
    "epochs": 20,
    "base_lr": 0.0005,
    "decay_lr": 0.7,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 2,
    "weight_decay": 3.6e-05,
    "use_mixed": False,
}
# 81.72\% $\pm$ 0.50
BEST_MINIMAL = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 32,
    "size": 320,
    "embed_dim": 64,
    "embed_dr": 0.0,
    "epochs": 20,
    "base_lr": 0.0005,
    "decay_lr": 0.8,
    "backbone_activation": "relu",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 1,
    "weight_decay": 0.00048,
    "use_mixed": False,
    "minimal": True,
}
# 61.76\% $\pm$ 6.14
BEST_LTC = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 32,
    "size": 128,
    "embed_dim": 64,
    "embed_dr": 0.0,
    "epochs": 20,
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "forget_bias": 2.4,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-05,
    "use_mixed": False,
    "minimal": False,
    "use_ltc": True,
}

def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plots
    plt.show()

def score(config):
    acc = []
    with open("results/training_results_imdb.txt", "w") as result_file:  # Mở file để lưu kết quả
        # for i in range(1):
        #     accuracy = 100 * eval(config, i)
        #     acc.append(accuracy)
        #     result_file.write(f"IMDB test accuracy [{len(acc)}/1]: {accuracy:.2f}%\n")
        #     print(f"IMDB test accuracy [{len(acc)}/1]: {np.mean(acc):0.2f}% ± {np.std(acc):0.2f}")
        # result_file.write(f"IMDB test accuracy: {np.mean(acc):0.2f}% ± {np.std(acc):0.2f}\n")
        history, accuracy = eval(config, 1)
        accuracy *= 100
        result_file.write(f"IMDB test accuracy: {accuracy:.2f}%\n")
        result_file.write(
            f"Training sparse_categorical_accuracy: {history['sparse_categorical_accuracy'][-1]:.4f}, "
            f"Training loss: {history['loss'][-1]:.4f}, "
            f"Validation sparse_categorical_accuracy: {history['val_sparse_categorical_accuracy'][-1]:.4f}, "
            f"Validation loss: {history['val_loss'][-1]:.4f}\n"
        )
    
    print(f"IMDB test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mixed", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--use_ltc", action="store_true")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=20)
    args = parser.parse_args()

    if args.minimal:
        BEST_MINIMAL["epochs"] = args.epochs
        score(BEST_MINIMAL)
    elif args.use_ltc:
        BEST_LTC["epochs"] = args.epochs
        score(BEST_LTC)
    else:
        BEST_DEFAULT["epochs"] = args.epochs
        score(BEST_DEFAULT)

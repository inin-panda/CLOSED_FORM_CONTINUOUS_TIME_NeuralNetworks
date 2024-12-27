import os
import sys
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from irregular_sampled_datasets import ETSMnistData
from tf_cfc import CfcCell, MixedCfcCell, LTCCell
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="cfc", choices=["cfc", "ltc"], help="Choose model type: 'cfc' or 'ltc'")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--lr", default=0.0005, type=float)
args = parser.parse_args()

data = ETSMnistData(time_major=False)

def create_cfc_model(data, size=64):
    CFC_CONFIG = {
        "backbone_activation": "gelu",
        "backbone_dr": 0.0,
        "forget_bias": 3.0,
        "backbone_units": 128,
        "backbone_layers": 1,
        "weight_decay": 0,
        "use_lstm": False,
        "no_gate": False,
        "minimal": False,
    }
    
    cell = CfcCell(units=size, hparams=CFC_CONFIG)
    
    pixel_input = tf.keras.Input(shape=(data.pad_size, 1), name="pixel")
    time_input = tf.keras.Input(shape=(data.pad_size, 1), name="time")
    mask_input = tf.keras.Input(shape=(data.pad_size,), dtype=tf.bool, name="mask")

    merged_input = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([pixel_input, time_input])
    rnn = tf.keras.layers.RNN(cell, return_sequences=False)
    output_states = rnn(merged_input, mask=mask_input)
    
    y = tf.keras.layers.Dense(10)(output_states)

    model = tf.keras.Model(inputs=[pixel_input, time_input, mask_input], outputs=[y])
    
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    
    return model

def create_ltc_model(data):
    pixel_input = tf.keras.Input(shape=(data.pad_size, 1), name="pixel")
    time_input = tf.keras.Input(shape=(data.pad_size, 1), name="time")
    mask_input = tf.keras.Input(shape=(data.pad_size,), dtype=tf.bool, name="mask")

    merged_input = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([pixel_input, time_input])
    rnn_ltc = tf.keras.layers.RNN(LTCCell(units=128), return_sequences=False)
    output_ltc = rnn_ltc(merged_input, mask=mask_input)
    dense_ltc = tf.keras.layers.Dense(10)(output_ltc)

    model_ltc = tf.keras.Model(inputs=[pixel_input, time_input, mask_input], outputs=[dense_ltc])

    model_ltc.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model_ltc

def plot_predictions(model, data, num_samples=5, model_name='CFC'):
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Randomly select samples
    sample_indices = np.random.choice(data.train_events.shape[0], num_samples)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    predictions = []

    # Predict and visualize
    for i, idx in enumerate(sample_indices):
        event_sample = data.train_events[idx:idx+1]
        elapsed_sample = data.train_elapsed[idx:idx+1]
        mask_sample = data.train_mask[idx:idx+1]

        # Predict with the model
        pred = model.predict([event_sample, elapsed_sample, mask_sample])
        predictions.append(pred)

        # Display image
        ax = axes[i]
        ax.imshow(data.train_events[idx].reshape(16, 16), cmap='gray')
        ax.set_title(f"Pred: {np.argmax(pred)}\nActual: {data.train_y[idx]}")
        ax.axis('off')

    # Save prediction results to a text file
    with open(f'results/{model_name}_predictions_et_smnist.txt', 'w') as f:
        f.write(f"Predictions for {model_name} Model:\n")
        for i in range(num_samples):
            line = f"Sample {i+1}: Predicted Label: {np.argmax(predictions[i])}, Actual Label: {data.train_y[sample_indices[i]]}\n"
            f.write(line)
            print(line.strip())  # Also print to console

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_predictions_et_smnist.png')
    plt.close()

def main():
    # Load data
    data = ETSMnistData(time_major=False)

    # CFC Model
    if args.model == "cfc":
        print("Training CFC Model...")
        cfc_model = create_cfc_model(data)
        cfc_model.summary()
        
        cfc_hist = cfc_model.fit(
            x=(data.train_events, data.train_elapsed, data.train_mask),
            y=data.train_y,
            batch_size=128,
            epochs=args.epochs,
        )

        _, cfc_test_acc = cfc_model.evaluate(
            x=(data.test_events, data.test_elapsed, data.test_mask), y=data.test_y
        )
        print(f"CFC Model Test Accuracy: {cfc_test_acc:.4f}")

        # Plot CFC predictions
        plot_predictions(cfc_model, data, model_name='CFC')

    elif args.model == "ltc":
        # LTC Model
        print("Training LTC Model...")
        ltc_model = create_ltc_model(data)
        ltc_model.summary()

        ltc_hist = ltc_model.fit(
            x=(data.train_events, data.train_elapsed, data.train_mask),
            y=data.train_y,
            batch_size=64,
            epochs=1,
        )

        _, ltc_test_acc = ltc_model.evaluate(
            x=(data.test_events, data.test_elapsed, data.test_mask),
            y=data.test_y
        )
        print(f"LTC Model Test Accuracy: {ltc_test_acc:.4f}")

        # Plot LTC predictions
        plot_predictions(ltc_model, data, model_name='LTC')

if __name__ == "__main__":
    main()
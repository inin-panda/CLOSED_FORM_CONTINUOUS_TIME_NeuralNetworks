import os
import subprocess

from irregular_sampled_datasets import Walker2dImitationData

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt 
from tf_cfc import CfcCell, MixedCfcCell, LTCCell
import sys



class BackupCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(BackupCallback, self).__init__()
        self.saved_weights = None
        self._model = model
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best_loss:
            self.best_loss = logs["val_loss"]
            print(f" new best -> {logs['val_loss']:0.3f}")
            self.saved_weights = self.model.get_weights()

    def restore(self):
        if self.best_loss is not None:
            self.model.set_weights(self.saved_weights)

def visualize_results(hist):
    # Vẽ đồ thị độ chính xác và mất mát
    plt.figure(figsize=(12, 4))

    # Đồ thị độ chính xác
    plt.subplot(1, 2, 1)
        # Kiểm tra xem key có tồn tại trong 'history' không
    if 'accuracy' in hist.history:
        plt.plot(hist.history['accuracy'], label='Train Accuracy')
    elif 'sparse_categorical_accuracy' in hist.history:
        plt.plot(hist.history['sparse_categorical_accuracy'], label='Train Accuracy')
    
    if 'val_accuracy' in hist.history:
        plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    elif 'val_sparse_categorical_accuracy' in hist.history:
        plt.plot(hist.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Đồ thị mất mát
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    

def eval(config, index_arg, verbose=0):
    data = Walker2dImitationData(seq_len=64)

    if config.get("use_ltc"):
        cell = LTCCell(units=config["size"])
    elif config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)
    else:
        cell = CfcCell(units=config["size"], hparams=config)

    signal_input = tf.keras.Input(shape=(data.seq_len, data.input_size), name="robot")
    time_input = tf.keras.Input(shape=(data.seq_len, 1), name="time")

    rnn = tf.keras.layers.RNN(cell, return_sequences=True)

    # output_states = rnn((signal_input, time_input))
    # Thay đổi cách concatenate input
    combined_input = tf.keras.layers.Concatenate()([signal_input, time_input])
    output_states = rnn(combined_input)
    
    y = tf.keras.layers.Dense(data.input_size)(output_states)

    model = tf.keras.Model(inputs=[signal_input, time_input], outputs=[y])

    base_lr = config["base_lr"]
    decay_lr = config["decay_lr"]
    train_steps = data.train_x.shape[0] // config["batch_size"]
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
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["accuracy"]  # Đảm bảo có đo lường độ chính xác
    )
    model.summary()

    # Fit model
    hist = model.fit(
        x=(data.train_x, data.train_times),
        y=data.train_y,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=((data.valid_x, data.valid_times), data.valid_y),
        callbacks=[BackupCallback(model)],
        verbose=0,
    )
    # Evaluate model after training
    test_loss , test_acc = model.evaluate(
        x=(data.test_x, data.test_times), y=data.test_y, verbose=2
    )
    mse = test_loss
    print(f"Test accuracy: {test_acc:.2f}%")  
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MSE: {mse:.4f}")# In độ chính xác trên test
    
    visualize_results(hist)
    return test_acc, test_loss, mse

# 0.64038 +- 0.00574
BEST_DEFAULT = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 256,
    "size": 64,
    "epochs": 2,
    "base_lr": 0.02,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 256,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
}

# MSE: 0.61654 +- 0.00634
BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "epochs": 20,
    "base_lr": 0.005,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.2,
    "forget_bias": 2.1,
    "backbone_units": 128,
    "backbone_layers": 2,
    "weight_decay": 6e-06,
    "use_mixed": True,
    "no_gate": False,
}

# 0.94844 $\pm$ 0.00988
BEST_MINIMAL = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 256,
    "epochs": 20,
    "base_lr": 0.006,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "forget_bias": 5.0,
    "backbone_units": 192,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
}

# 0.66225 $\pm$ 0.01330
BEST_LTC = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
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
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}
  
def score(config):
    acc = []
    losses = []
    mse_values = []

    with open("training_Walker_results.txt", "w") as result_file:  # Mở file để lưu kết quả
        for i in range(1):
            accuracy, loss, mse = eval(config, i)
            acc.append(accuracy)
            losses.append(loss)
            mse_values.append(mse)

            result_file.write(f"Test Accuracy [{i+1}/2]: {accuracy:.2f}%\n")
            result_file.write(f"Test Loss [{i+1}/2]: {loss:.4f}\n")
            result_file.write(f"Test MSE [{i+1}/2]: {mse:.4f}\n")
            print(f"Test Accuracy after iteration {i+1}: {accuracy:.2f}%")
            print(f"Test Loss after iteration {i+1}: {loss:.4f}")
            print(f"Test MSE after iteration {i+1}: {mse:.4f}")
        
        # In ra độ chính xác, loss và MSE trung bình của 5 lần chạy
        mean_accuracy = np.mean(acc)
        std_accuracy = np.std(acc)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        mean_mse = np.mean(mse_values)
        std_mse = np.std(mse_values)

        result_file.write(f"Average Test Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%\n")
        result_file.write(f"Average Test Loss: {mean_loss:.4f} ± {std_loss:.4f}\n")
        result_file.write(f"Average Test MSE: {mean_mse:.4f} ± {std_mse:.4f}\n")

        print(f"Average Test Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%")
        print(f"Average Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"Average Test MSE: {mean_mse:.4f} ± {std_mse:.4f}")  


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


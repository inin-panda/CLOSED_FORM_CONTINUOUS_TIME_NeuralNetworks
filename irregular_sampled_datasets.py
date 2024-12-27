# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm


class Walker2dImitationData:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        os.makedirs("data", exist_ok=True)
        if not os.path.isfile("data/walker/rollout_000.npy"):
            os.system("wget https://people.csail.mit.edu/mlechner/datasets/walker.zip")
            os.system("unzip walker.zip -d data/")
        all_files = sorted(
            [
                os.path.join("data/walker", d)
                for d in os.listdir("data/walker")
                if d.endswith(".npy")
            ]
        )

        self.rng = np.random.RandomState(891374)
        np.random.RandomState(125487).shuffle(all_files)
        # 15% test set, 10% validation set, the rest is for training
        test_n = int(0.15 * len(all_files))
        valid_n = int((0.15 + 0.1) * len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        train_x, train_t, train_y = self._load_files(train_files)
        valid_x, valid_t, valid_y = self._load_files(valid_files)
        test_x, test_t, test_y = self._load_files(test_files)

        train_x, train_t, train_y = self.perturb_sequences(train_x, train_t, train_y)
        valid_x, valid_t, valid_y = self.perturb_sequences(valid_x, valid_t, valid_y)
        test_x, test_t, test_y = self.perturb_sequences(test_x, test_t, test_y)

        self.train_x, self.train_times, self.train_y = self.align_sequences(
            train_x, train_t, train_y
        )
        self.valid_x, self.valid_times, self.valid_y = self.align_sequences(
            valid_x, valid_t, valid_y
        )
        self.test_x, self.test_times, self.test_y = self.align_sequences(
            test_x, test_t, test_y
        )
        self.input_size = self.train_x.shape[-1]

        # print("train_times: ", str(self.train_times.shape))
        # print("train_x: ", str(self.train_x.shape))
        # print("train_y: ", str(self.train_y.shape))

    def align_sequences(self, set_x, set_t, set_y):

        times = []
        x = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_t = set_t[i]
            seq_y = set_y[i]

            for t in range(0, seq_y.shape[0] - self.seq_len, self.seq_len // 4):
                x.append(seq_x[t : t + self.seq_len])
                times.append(seq_t[t : t + self.seq_len])
                y.append(seq_y[t : t + self.seq_len])

        return (
            np.stack(x, axis=0),
            np.expand_dims(np.stack(times, axis=0), axis=-1),
            np.stack(y, axis=0),
        )

    def perturb_sequences(self, set_x, set_t, set_y):

        x = []
        times = []
        y = []
        for i in range(len(set_y)):

            seq_x = set_x[i]
            seq_y = set_y[i]

            new_x, new_times = [], []
            new_y = []

            skip = 0
            for t in range(seq_y.shape[0]):
                skip += 1
                if self.rng.rand() < 0.9:
                    new_x.append(seq_x[t])
                    new_times.append(skip)
                    new_y.append(seq_y[t])
                    skip = 0

            x.append(np.stack(new_x, axis=0))
            times.append(np.stack(new_times, axis=0))
            y.append(np.stack(new_y, axis=0))

        return x, times, y

    def _load_files(self, files):
        all_x = []
        all_t = []
        all_y = []
        for f in files:

            arr = np.load(f)
            x_state = arr[:-1, :].astype(np.float32)
            y = arr[1:, :].astype(np.float32)

            x_times = np.ones(x_state.shape[0])
            all_x.append(x_state)
            all_t.append(x_times)
            all_y.append(y)

            # print("Loaded file '{}' of length {:d}".format(f, x_state.shape[0]))
        return all_x, all_t, all_y


class ETSMnistData:
    def __init__(self, time_major, pad_size=256):
        self.threshold = 128
        self.pad_size = pad_size

        if not self.load_from_cache():
            self.create_dataset()

        self.train_elapsed /= self.pad_size
        self.test_elapsed /= self.pad_size

    def load_from_cache(self):
        if os.path.isfile("dataset/test_mask.npy"):
            self.train_events = np.load("dataset/train_events.npy")
            self.train_elapsed = np.load("dataset/train_elapsed.npy")
            self.train_mask = np.load("dataset/train_mask.npy")
            self.train_y = np.load("dataset/train_y.npy")

            self.test_events = np.load("dataset/test_events.npy")
            self.test_elapsed = np.load("dataset/test_elapsed.npy")
            self.test_mask = np.load("dataset/test_mask.npy")
            self.test_y = np.load("dataset/test_y.npy")

            print("train_events.shape: ", str(self.train_events.shape))
            print("train_elapsed.shape: ", str(self.train_elapsed.shape))
            print("train_mask.shape: ", str(self.train_mask.shape))
            print("train_y.shape: ", str(self.train_y.shape))

            print("test_events.shape: ", str(self.test_events.shape))
            print("test_elapsed.shape: ", str(self.test_elapsed.shape))
            print("test_mask.shape: ", str(self.test_mask.shape))
            print("test_y.shape: ", str(self.test_y.shape))
            return True
        return False

    def transform_sample(self, x):
        x = x.flatten()

        events = np.zeros([self.pad_size, 1], dtype=np.float32)
        elapsed = np.zeros([self.pad_size, 1], dtype=np.float32)
        mask = np.zeros([self.pad_size], dtype=np.bool)

        last_char = -1
        write_index = 0
        elapsed_counter = 0
        for i in range(x.shape[0]):
            elapsed_counter += 1
            char = int(x[i] > self.threshold)
            if last_char != char:
                events[write_index] = char
                elapsed[write_index] = elapsed_counter
                mask[write_index] = True
                write_index += 1
                if write_index >= self.pad_size:
                    # Enough 1s in this sample, abort
                    self._abort_counter += 1
                    break
                elapsed_counter = 0
            last_char = char
        self._all_lenghts.append(write_index)
        return events, elapsed, mask

    def transform_array(self, x):
        events_list = []
        elapsed_list = []
        mask_list = []

        for i in tqdm(range(x.shape[0])):
            events, elapsed, mask = self.transform_sample(x[i])
            events_list.append(events)
            elapsed_list.append(elapsed)
            mask_list.append(mask)

        return (
            np.stack(events_list, axis=0),
            np.stack(elapsed_list, axis=0),
            np.stack(mask_list, axis=0),
        )

    def create_dataset(self):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

        self._all_lenghts = []
        self._abort_counter = 0

        train_x = train_x.reshape([-1, 28 * 28])
        test_x = test_x.reshape([-1, 28 * 28])

        self.train_y = train_y
        self.test_y = test_y

        print("Transforming training samples")
        self.train_events, self.train_elapsed, self.train_mask = self.transform_array(
            train_x
        )
        print("Transforming test samples")
        self.test_events, self.test_elapsed, self.test_mask = self.transform_array(
            test_x
        )

        print("Average time-series length: {:0.2f}".format(np.mean(self._all_lenghts)))
        print("Abort counter: ", str(self._abort_counter))
        os.makedirs("dataset", exist_ok=True)
        np.save("dataset/train_events.npy", self.train_events)
        np.save("dataset/train_elapsed.npy", self.train_elapsed)
        np.save("dataset/train_mask.npy", self.train_mask)
        np.save("dataset/train_y.npy", self.train_y)

        np.save("dataset/test_events.npy", self.test_events)
        np.save("dataset/test_elapsed.npy", self.test_elapsed)
        np.save("dataset/test_mask.npy", self.test_mask)
        np.save("dataset/test_y.npy", self.test_y)


class XORData:
    def __init__(self, time_major, event_based=True, pad_size=24):
        self.pad_size = pad_size
        self.event_based = event_based
        self._abort_counter = 0
        if not self.load_from_cache():
            self.create_dataset()

        self.train_elapsed /= self.pad_size
        self.test_elapsed /= self.pad_size

    def load_from_cache(self):
        if os.path.isfile("dataset/xor_test_y.npy"):
            self.train_events = np.load("dataset/xor_train_events.npy")
            self.train_elapsed = np.load("dataset/xor_train_elapsed.npy")
            self.train_mask = np.load("dataset/xor_train_mask.npy")
            self.train_y = np.load("dataset/xor_train_y.npy")

            self.test_events = np.load("dataset/xor_test_events.npy")
            self.test_elapsed = np.load("dataset/xor_test_elapsed.npy")
            self.test_mask = np.load("dataset/xor_test_mask.npy")
            self.test_y = np.load("dataset/xor_test_y.npy")

            print("train_events.shape: ", str(self.train_events.shape))
            print("train_elapsed.shape: ", str(self.train_elapsed.shape))
            print("train_mask.shape: ", str(self.train_mask.shape))
            print("train_y.shape: ", str(self.train_y.shape))

            print("test_events.shape: ", str(self.test_events.shape))
            print("test_elapsed.shape: ", str(self.test_elapsed.shape))
            print("test_mask.shape: ", str(self.test_mask.shape))
            print("test_y.shape: ", str(self.test_y.shape))
            return True
        return False

    def create_event_based_sample(self, rng):

        label = 0
        events = np.zeros([self.pad_size, 1], dtype=np.float32)
        elapsed = np.zeros([self.pad_size, 1], dtype=np.float32)
        mask = np.zeros([self.pad_size], dtype=np.bool)

        last_char = -1
        write_index = 0
        elapsed_counter = 0
        length = rng.randint(low=2, high=self.pad_size)

        for i in range(length):
            elapsed_counter += 1

            char = int(rng.randint(low=0, high=2))
            label += char
            if last_char != char:
                events[write_index] = char
                elapsed[write_index] = elapsed_counter
                mask[write_index] = True
                write_index += 1
                elapsed_counter = 0
                if write_index >= self.pad_size - 1:
                    # Enough 1s in this sample, abort
                    self._abort_counter += 1
                    break
            last_char = char
        if elapsed_counter > 0:
            events[write_index] = char
            elapsed[write_index] = elapsed_counter
            mask[write_index] = True
        label = label % 2
        return events, elapsed, mask, label

    def create_dense_sample(self, rng):

        label = 0
        events = np.zeros([self.pad_size, 1], dtype=np.float32)
        elapsed = np.zeros([self.pad_size, 1], dtype=np.float32)
        mask = np.zeros([self.pad_size], dtype=np.bool)

        last_char = -1
        write_index = 0
        elapsed_counter = 0

        length = rng.randint(low=2, high=self.pad_size)
        for i in range(length):
            elapsed_counter += 1

            char = int(rng.randint(low=0, high=2))
            label += char
            events[write_index] = char
            elapsed[write_index] = elapsed_counter
            mask[write_index] = True
            write_index += 1
            elapsed_counter = 0
        label = label % 2
        label2 = int(np.sum(events)) % 2
        assert label == label2
        return events, elapsed, mask, label

    def create_set(self, size, seed):
        rng = np.random.RandomState(seed)
        events_list = []
        elapsed_list = []
        mask_list = []
        label_list = []

        for i in tqdm(range(size)):
            if self.event_based:
                events, elapsed, mask, label = self.create_event_based_sample(rng)
            else:
                events, elapsed, mask, label = self.create_dense_sample(rng)
            events_list.append(events)
            elapsed_list.append(elapsed)
            mask_list.append(mask)
            label_list.append(label)

        return (
            np.stack(events_list, axis=0),
            np.stack(elapsed_list, axis=0),
            np.stack(mask_list, axis=0),
            np.stack(label_list, axis=0),
        )

    def create_dataset(self):

        print("Transforming training samples")
        (
            self.train_events,
            self.train_elapsed,
            self.train_mask,
            self.train_y,
        ) = self.create_set(100000, 1234984)
        print("Transforming test samples")
        (
            self.test_events,
            self.test_elapsed,
            self.test_mask,
            self.test_y,
        ) = self.create_set(10000, 48736)

        print("train_events.shape: ", str(self.train_events.shape))
        print("train_elapsed.shape: ", str(self.train_elapsed.shape))
        print("train_mask.shape: ", str(self.train_mask.shape))
        print("train_y.shape: ", str(self.train_y.shape))

        print("test_events.shape: ", str(self.test_events.shape))
        print("test_elapsed.shape: ", str(self.test_elapsed.shape))
        print("test_mask.shape: ", str(self.test_mask.shape))
        print("test_y.shape: ", str(self.test_y.shape))

        print("Abort counter: ", str(self._abort_counter))
        os.makedirs("dataset", exist_ok=True)
        np.save("dataset/xor_train_events.npy", self.train_events)
        np.save("dataset/xor_train_elapsed.npy", self.train_elapsed)
        np.save("dataset/xor_train_mask.npy", self.train_mask)
        np.save("dataset/xor_train_y.npy", self.train_y)

        np.save("dataset/xor_test_events.npy", self.test_events)
        np.save("dataset/xor_test_elapsed.npy", self.test_elapsed)
        np.save("dataset/xor_test_mask.npy", self.test_mask)
        np.save("dataset/xor_test_y.npy", self.test_y)


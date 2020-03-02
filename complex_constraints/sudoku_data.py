import numpy as np
from numpy.random import permutation


class SudokuData():
    def __init__(self, data_path):
        np.random.seed(0)
        labels = []
        data = []
        with open(data_path) as file:
            for line in file:
                linesplit = line.strip().split(';')
                # Doesn't have enough entries, isn't data
                if len(linesplit) != 2:
                    continue
                labels.append(linesplit[0])
                data.append(linesplit[1])

        # We're going to split 60/20/20 train/valid/test
        # Nonrandom version
        perm = permutation(len(data))
        train_inds = perm[:int(len(data) * 0.8)]
        valid_inds = perm[int(len(data) * 0.8):int(len(data) * 0.9)]
        test_inds = perm[int(len(data) * 0.9):]
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.train_data = self.data[train_inds, :]
        self.valid_data = self.data[valid_inds, :]
        self.test_data = self.data[test_inds, :]
        self.train_labels = self.labels[train_inds, :]
        self.valid_labels = self.labels[valid_inds, :]
        self.test_labels = self.labels[test_inds, :]

        self.batch_ind = len(train_inds)
        self.batch_perm = None
        np.random.seed()

    def get_batch(self, size):
        # If we're out:
        if self.batch_ind >= self.train_data.shape[0]:
            # Rerandomize ordering
            self.batch_perm = permutation(self.train_data.shape[0])
            # Reset counter
            self.batch_ind = 0

        # If there's not enough
        if self.train_data.shape[0] - self.batch_ind < size:
            # Get what there is, append whatever else you need
            ret_ind = self.batch_perm[self.batch_ind:]
            d, l = self.train_data[ret_ind, :], self.train_labels[ret_ind, :]
            size -= len(ret_ind)
            self.batch_ind = self.train_data.shape[0]
            nd, nl = self.get_batch(size)
            return np.concatenate(d, nd), np.concatenate(l, nl)

        # Normal case
        ret_ind = self.batch_perm[self.batch_ind: self.batch_ind + size]
        return self.train_data[ret_ind, :], self.train_labels[ret_ind, :]


def to_one_hot(val):
    one_hot = np.zeros(4)
    if val > 0:
        one_hot[val - 1] = 1
    return one_hot


if __name__ == '__main__':
    s = SudokuData('sushi.soc')

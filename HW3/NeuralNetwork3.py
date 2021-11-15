import numpy as np
import math
import sys


class NeuralNetwork:

    def __init__(self, inputs, h1, h2, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.h1 = h1
        self.h2 = h2

    def feed_f(self):
        pass

    def back_p(self):
        pass


def read_images(file_name):
    return 1


def read_labels(file_name):
    return 1


def write():
    pass


def get_batches():
    pass


if __name__ == "__main__":
    train_inputs = read_images(sys.argv[0])
    train_labels = read_labels(sys.argv[1])
    test_inputs = read_images(sys.argv[2])

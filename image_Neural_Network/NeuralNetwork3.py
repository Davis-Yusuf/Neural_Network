import numpy as np
import math
import argparse
import sys


def activation_function():
    pass


class NeuralNetwork:

    def __init__(self, inputs, h1, h2, out):
        self.inputs = inputs
        self.h1 = h1
        self.h2 = h2
        self.out = out
        self.weights_i_h1 = np.random.uniform(-1, 1, (128, 784))
        self.weights_h1_h2 = np.random.uniform(-1, 1, (128, 128))
        self.weights_h2_o = np.random.uniform(-1, 1, (10, 128))
        self.out_bias = np.ones(10)
        self.h1_bias = np.ones(128)
        self.h2_bias = np.ones(128)

    def feed_f(self, inputs, targets):
        self.inputs = inputs
        output_output =
        h1_ouput = {}
        h2_output = {}
        for i in range(128):
            h1_ouput[i] = np.sum(np.matmul(inputs, self.weights_i_h1))

    def back_p(self):
        pass


def read(file_name):
    with open(file_name) as file:
        dataset = np.genfromtxt(file, delimiter=",")

    print(dataset.shape)
    return dataset


# def read_labels(file_name):
#     with open(file_name) as file:
#         dataset = file.readlines()
#         data = [i.rstrip('\n') for i in dataset]
#
#     return data


def write():
    pass


def get_batches(dataset):  # Currently assuming that the size will be perfectly divisible
    size = dataset.shape
    batches = []
    if size[0] >= 10000:
        if size[0] % 100 == 0:
            start = 0
            end = int(size[0] / 100)
            for _ in range(100):
                batches.append(np.array(dataset[start:end]))
                start = end
                end = end + int(size[0] / 100)
        # else:
        #     start = 0
        #     end = int(math.floor(size[0]/100))
        #     for i in range(100):
        #         batches.append(np.array(dataset[start:end]))
        #         start = end
        #         end = end + int(size[0]/100)
    elif size[0] >= 1000:
        if size[0] % 10 == 0:
            start = 0
            end = int(size[0] / 10)
            for _ in range(10):
                batches.append(np.array(dataset[start:end]))
                start = end
                end = end + int(size[0] / 10)
    # else:
    #     pass

    # arr = batches[:12]
    # n_arr = np.vstack(arr)
    # np.savetxt('training_sample_label.csv', n_arr, delimiter=',', fmt='%d')
    return batches


if __name__ == "__main__":
    train_inputs = read(sys.argv[1])
    train_labels = read(sys.argv[2])
    # test_inputs = read(sys.argv[3])
    my_agent = NeuralNetwork(1, 2, 3, 4)
    print(my_agent.out_bias)
    all_batches = get_batches(train_inputs)

    for i in range(len(all_batches)):
        temp_batches = all_batches[i]
        for j in range(all_batches[0].shape[0]):
            input_nodes = temp_batches[j] / 255
            # my_agent.inputs = input_nodes
            # my_agent.feed_f()


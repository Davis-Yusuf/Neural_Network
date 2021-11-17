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
        self.weights_i_h1 = [0] * inputs
        self.weights_h1_h2 = [0] * h1
        self.weights_h2_o = [0] * h2
        self.h2_out_bias = [[0] * 1 for _ in range(h2)]
        self.in_h1_bias = [[0] * 1 for _ in range(inputs)]
        self.h1_h2_bias = [[0] * 1 for _ in range(h1)]
        for k in range(inputs):
            self.weights_i_h1[k] = np.random.uniform(-1, 1, inputs)
            self.in_h1_bias[k] = np.ones(inputs)
        for z in range(h1):
            self.weights_h1_h2[z] = np.random.uniform(-1, 1, h2)
            self.h1_h2_bias[z] = np.ones(h2)
        for c in range(out):
            self.weights_h2_o[c] = np.random.uniform(-1, 1, out)
            self.h2_out_bias[c] = np.ones(out)

    def feed_f(self, inputs, targets):

        h1_output = {}
        h2_output = {}
        out_output = {}

        # print(inputs[1].shape, self.weights_i_h1[1].shape)
        for i in range(self.h1):
            h1_output[i] = np.sum(np.matmul(inputs[i] / 255, self.weights_i_h1[i]))
            h1_output[i] = np.add(h1_output[i], self.in_h1_bias[i])

        print(h1_output)
        print(h1_output[1])


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
    my_agent = NeuralNetwork(784, 128, 128, 10)
    my_agent.feed_f(train_inputs, 2)
    # all_batches = get_batches(train_inputs)
    #
    # for i in range(len(all_batches)):
    #     temp_batches = all_batches[i]
    #     for j in range(all_batches[0].shape[0]):
    #         input_nodes = temp_batches[j] / 255
    #         # my_agent.inputs = input_nodes
    #         # my_agent.feed_f()


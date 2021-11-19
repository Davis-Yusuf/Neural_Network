import numpy as np
import math
import sys


def sig_activation_function(node):
    result = 1 / (1 + math.exp(-node))
    return result


def sig_derivative(node):
    result = sig_activation_function(node)*(1 - sig_activation_function(node))
    return result


def softmax_activation_function(arr):
    result = [0] * 10
    for jj in range(10):
        result[jj] = np.exp(arr[jj]) / np.sum(np.exp(arr))
    return result


def cross_entropy_func(labels, predictions):
    result = [0] * 10
    for ii in range(10):
        result[ii] = -np.sum(labels[ii], math.log2(predictions[ii]))  #P,q are probability distribution

    return result


class NeuralNetwork:

    def __init__(self, inputs, h1, h2, out):
        self.inputs = inputs
        self.h1 = h1
        self.h2 = h2
        self.out = out
        self.weights_i_h1 = np.random.uniform(-1, 1, inputs)
        self.weights_h1_h2 = np.random.uniform(-1, 1, h2)
        self.weights_h2_o = np.random.uniform(-1, 1, h2)
        self.h2_bias = np.ones(h2)
        self.h1_bias = np.ones(h1)
        self.out_bias = np.ones(out)
        for k in range(h1 - 1):
            self.weights_i_h1 = np.vstack((self.weights_i_h1, np.random.uniform(-1, 1, inputs)))
        for z in range(h2 - 1):
            self.weights_h1_h2 = np.vstack((self.weights_h1_h2, np.random.uniform(-1, 1, h2)))
        for r in range(out - 1):
            self.weights_h2_o = np.vstack((self.weights_h2_o, np.random.uniform(-1, 1, h2)))

    def feed_f(self, inputs, targets):

        h1_output =
        z1_output = {}
        h2_output = {}
        z2_output = {}
        out_output = {}
        label_list = np.zeros(self.out)

        for i in range(self.h1):
            z1_output[i] = np.sum(np.matmul(inputs / 255, self.weights_i_h1[i])) + self.h1_bias[i]
            h1_output[i] = sig_activation_function(z1_output[i])

        h1_nodes = list(h1_output.values())
        arr = np.array(h1_nodes)
        print(arr.shape)

        for j in range(self.h2):
            z2_output[j] = np.sum(np.matmul(arr, self.weights_h1_h2[j])) + self.h2_bias[j]
            h2_output[j] = sig_activation_function(z2_output[j])

        h2_nodes = list(h2_output.values())
        arr2 = np.array(h2_nodes)

        for k in range(self.out):
            out_output[k] = np.sum(np.matmul(arr2, self.weights_h2_o[k])) + self.out_bias[k]
            label_list[int(targets)] = 1
            # out_output[k] = softmax_activation_function(out_output[k])

        out_nodes = list(out_output.values())
        arr3 = np.array(out_nodes)
        print(label_list)
        final = softmax_activation_function(arr3)
        # temp = cross_entropy_func(label_list, final

        z1_output = list(z1_output.values())
        arr_z1 = np.array(z1_output)
        z2_output = list(z2_output.values())
        arr_z2 = np.array(z2_output)

        return final, label_list, arr_z1, arr_z2, arr, arr2, inputs

    def back_p(self, out_softmax, labels, z1, z2, arr, arr2, inputs, learning_rate):
        costs = {}

        out_grad = np.subtract(out_softmax, labels)
        h2_a_grad = np.dot(np.transpose(self.weights_h2_o), out_grad)
        h2_w_grad = learning_rate * np.dot(np.transpose(arr2), out_grad)
        h2_b_grad = out_grad
        self.weights_h2_o += h2_w_grad
        self.out_bias += h2_b_grad

        h2_grad = np.multiply(h2_a_grad, sig_derivative(z2))
        h1_a_grad = np.dot(h2_grad, np.transpose(self.weights_h1_h2))
        h1_w_grad = learning_rate * np.dot(np.transpose(arr), h2_grad)
        h1_b_grad = h2_grad
        self.weights_h1_h2 += h1_w_grad
        self.h2_bias += h1_b_grad

        h1_grad = np.multiply(h1_a_grad, sig_derivative(z1))
        in_a_grad = np.dot(h1_grad, np.transpose(self.weights_i_h1))
        in_w_grad = np.dot(np.transpose(inputs), h1_grad)
        in_b_grad = h1_grad
        self.weights_i_h1 += in_w_grad
        self.h1_bias += in_b_grad


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
    print(type(train_inputs))
    print(type(train_inputs[1]))
    a, b, c, d, e, f, g = my_agent.feed_f(train_inputs[5], train_labels[5])
    my_agent.back_p(a, b, c, d, e, f, g, 0.1)
    print('done')

    # all_batches = get_batches(train_inputs)
    #
    # for i in range(len(all_batches)):
    #     temp_batches = all_batches[i]
    #     for j in range(all_batches[0].shape[0]):
    #         input_nodes = temp_batches[j] / 255
    #         # my_agent.inputs = input_nodes
    #         # my_agent.feed_f()


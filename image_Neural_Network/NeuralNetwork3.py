import numpy as np
import math
import sys
import pandas as pd


def sig_activation_function(node):
    result = 1 / (1 + np.exp(-node))
    return result


def sig_derivative(node):
    result = sig_activation_function(node) * (1 - sig_activation_function(node))
    return result


def softmax_activation_function(arr):
    result = np.divide(np.exp(arr[0]), np.sum(np.exp(arr)))
    for jj in range(1, 10):
        result = np.vstack((result, np.divide(np.exp(arr[jj]), np.sum(np.exp(arr)))))
    return result


def cross_entropy_func(labels, predictions):
    result = [0] * 10
    for ii in range(10):
        result[ii] = -np.sum(labels[ii], math.log2(predictions[ii]))

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
        self.h2_bias = np.zeros((h2, 1))
        self.h1_bias = np.zeros((h1, 1))
        self.out_bias = np.zeros((out, 1))
        for _ in range(h1 - 1):
            self.weights_i_h1 = np.vstack((self.weights_i_h1, np.random.uniform(-1, 1, inputs)))
        for _ in range(h2 - 1):
            self.weights_h1_h2 = np.vstack((self.weights_h1_h2, np.random.uniform(-1, 1, h2)))
        for _ in range(out - 1):
            self.weights_h2_o = np.vstack((self.weights_h2_o, np.random.uniform(-1, 1, h2)))

    def feed_f(self, inputs):

        z1_output = np.sum(np.matmul(inputs / 255, self.weights_i_h1[0])) + self.h1_bias[0]
        h1_output = sig_activation_function(z1_output)

        for i in range(1, self.h1):
            z1_output = np.vstack((z1_output, np.sum(np.matmul(inputs / 255, self.weights_i_h1[i])) + self.h1_bias[i]))
            h1_output = np.vstack((h1_output, sig_activation_function(z1_output[i])))

        z2_output = np.sum(np.matmul(np.transpose(h1_output), self.weights_h1_h2[0])) + self.h2_bias[0]
        h2_output = sig_activation_function(z2_output)
        for j in range(1, self.h2):
            z2_output = np.vstack((z2_output, np.sum(np.matmul(np.transpose(h1_output), self.weights_h1_h2[j])) + self.h2_bias[j]))
            h2_output = np.vstack((h2_output, sig_activation_function(z2_output[j])))

        out_output = np.sum(np.matmul(np.transpose(h2_output), self.weights_h2_o[0])) + self.out_bias[0]
        for k in range(1, self.out):
            out_output = np.vstack((out_output, np.sum(np.matmul(np.transpose(h2_output), self.weights_h2_o[k])) + self.out_bias[k]))

        final = softmax_activation_function(out_output)
        # temp = cross_entropy_func(label_list, final

        return final, z1_output, z2_output, h1_output, h2_output, inputs

    def back_p(self, out_softmax, target, z1, z2, arr, arr2, inputs, learning_rate):
        label_list = np.zeros(self.out)
        label_list[int(target)] = 1
        new_labels = label_list.reshape((10, 1))

        out_grad = np.subtract(out_softmax, new_labels)
        h2_a_grad = np.dot(np.transpose(self.weights_h2_o), out_grad)
        h2_w_grad = learning_rate * np.dot(out_grad, np.transpose(arr2))
        h2_b_grad = out_grad

        h2_grad = np.multiply(h2_a_grad, sig_derivative(z2))
        h1_a_grad = np.dot(np.transpose(self.weights_h1_h2), h2_grad)
        h1_w_grad = learning_rate * np.dot(h2_grad, np.transpose(arr))
        h1_b_grad = h2_grad

        h1_grad = np.multiply(h1_a_grad, sig_derivative(z1))
        # in_a_grad = np.dot(h1_grad, np.transpose(self.weights_i_h1))
        new_inputs = inputs.reshape((784, 1))
        in_w_grad = learning_rate * np.dot(h1_grad, np.transpose(new_inputs))
        in_b_grad = h1_grad

        return h2_w_grad, h2_b_grad, h1_w_grad, h1_b_grad, in_w_grad, in_b_grad


def read(file_name):
    with open(file_name) as file:
        dataset = np.genfromtxt(file, delimiter=",")

    return dataset


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
    return batches


if __name__ == "__main__":
    train_inputs = read(sys.argv[1])
    train_labels = read(sys.argv[2])
    test_inputs = read(sys.argv[3])
    test_labels = read(sys.argv[4])

    # # training = train_inputs[:5200]
    # testing = train_inputs[5200:]
    # # training_l = train_labels[:5200]
    # testing_l = train_labels[5200:]

    # total = testing.shape[0]
    epochs = 20
    my_agent = NeuralNetwork(784, 256, 256, 10)

    h2_out_weight = np.zeros((10, 256))
    h1_h2_weight = np.zeros((256, 256))
    in_h1_weight = np.zeros((256, 784))
    h2_out_bias = np.zeros((10, 1))
    h1_h2_bias = np.zeros((256, 1))
    in_h1_bias = np.zeros((256, 1))

    train_batches = get_batches(train_inputs)
    label_batches = get_batches(train_labels)

    for e in range(epochs):
        for g in range(len(train_batches)):
            temp_batches = train_batches[g]
            temp2_batches = label_batches[g]
            for h in range(temp_batches.shape[0]):
                a2, b2, c2, d2, e2, f2 = my_agent.feed_f(temp_batches[h])
                temp_h2_w, temp_h2_b, temp_h1_w, temp_h1_b, temp_in_w, temp_in_b = my_agent.back_p(a2, temp2_batches[h], b2, c2, d2, e2, f2, 0.01)
                h2_out_weight = h2_out_weight + temp_h2_w
                h1_h2_weight = h1_h2_weight + temp_h1_w
                in_h1_weight = in_h1_weight + temp_in_w
                h2_out_bias = h2_out_bias + temp_h2_b
                h1_h2_bias = h1_h2_bias + temp_h1_b
                in_h1_bias = in_h1_bias + temp_in_b

            my_agent.weights_i_h1 = my_agent.weights_i_h1 + (in_h1_weight / temp_batches.shape[0])
            my_agent.weights_h1_h2 = my_agent.weights_h1_h2 + (h1_h2_weight / temp_batches.shape[0])
            my_agent.weights_h2_o = my_agent.weights_h2_o + (h2_out_weight / temp_batches.shape[0])

            my_agent.h2_bias = my_agent.h2_bias + (in_h1_bias / temp_batches.shape[0])
            my_agent.h1_bias = my_agent.h1_bias + (h1_h2_bias / temp_batches.shape[0])
            my_agent.out_bias = my_agent.out_bias + (h2_out_bias / temp_batches.shape[0])

    correct = 0
    list_1 = []
    for n in range(test_inputs.shape[0]):
        a1, c1, d1, e1, f1, g1 = my_agent.feed_f(test_inputs[n])
        list_1.append(np.argmax(a1))
        answer = np.argmax(a1)
        if answer == int(test_labels[n]):
            correct = correct + 1
    print(100 * (correct / test_inputs.shape[0]))

    # answer = np.array(list_1)
    # pd.DataFrame(answer).to_csv('test_predictions.csv', header=False, index=False)

    print('done')


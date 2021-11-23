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


def softmax_activation_function(node, arr):
    result = np.exp(node) / np.sum(np.exp(arr))
    # result = np.divide(np.exp(arr[0]), np.sum(np.exp(arr)))
    # for jj in range(1, 10):
    #     result = np.vstack((result, np.divide(np.exp(arr[jj]), np.sum(np.exp(arr)))))
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
        self.weights_i_h1 = np.random.uniform(-1, 1, (h2, inputs))
        self.weights_h1_h2 = np.random.uniform(-1, 1, (h2, h2))
        self.weights_h2_o = np.random.uniform(-1, 1, (out, h2))
        self.h2_bias = np.zeros((h2, 1))
        self.h1_bias = np.zeros((h1, 1))
        self.out_bias = np.zeros((out, 1))
        # for _ in range(h1 - 1):
        #     self.weights_i_h1 = np.vstack((self.weights_i_h1, np.random.uniform(-1, 1, inputs)))
        # for _ in range(h2 - 1):
        #     self.weights_h1_h2 = np.vstack((self.weights_h1_h2, np.random.uniform(-1, 1, h2)))
        # for _ in range(out - 1):
        #     self.weights_h2_o = np.vstack((self.weights_h2_o, np.random.uniform(-1, 1, h2)))

    def feed_f(self, inputs):

        z1_output = self.weights_i_h1.dot(inputs / 255) + self.h1_bias
        h1_output = np.array([sig_activation_function(z1_output[p]) for p in range(z1_output.shape[0])])

        # for i in range(1, self.h1):
        #     z1_output = np.vstack((z1_output, np.sum(np.dot(inputs / 255, self.weights_i_h1[i])) + self.h1_bias[i]))
        #     h1_output = np.vstack((h1_output, sig_activation_function(z1_output[i])))

        # z2_output = np.sum(np.dot(np.transpose(h1_output), self.weights_h1_h2[0])) + self.h2_bias[0]
        z2_output = self.weights_h1_h2.dot(h1_output) + self.h2_bias
        # h2_output = sig_activation_function(z2_output)
        h2_output = np.array([sig_activation_function(z2_output[p]) for p in range(z2_output.shape[0])])

        # for j in range(1, self.h2):
        #     z2_output = np.vstack((z2_output, np.sum(np.dot(np.transpose(h1_output), self.weights_h1_h2[j])) + self.h2_bias[j]))
        #     h2_output = np.vstack((h2_output, sig_activation_function(z2_output[j])))

        # out_output = np.sum(np.dot(np.transpose(h2_output), self.weights_h2_o[0])) + self.out_bias[0]
        out_output = self.weights_h2_o.dot(h2_output) + self.out_bias
        # for k in range(1, self.out):
        #     out_output = np.vstack((out_output, np.sum(np.dot(np.transpose(h2_output), self.weights_h2_o[k])) + self.out_bias[k]))
        # final = softmax_activation_function(out_output)
        final = np.array([softmax_activation_function(out_output[p], out_output) for p in range(out_output.shape[0])])
        # temp = cross_entropy_func(label_list, final

        return final, z1_output, z2_output, h1_output, h2_output, inputs

    def back_p(self, out_softmax, targets, z1, z2, arr, arr2, inputs):
        label_list = np.zeros((self.out, inputs.shape[1]))
        for h in range(label_list.shape[1]):
            label_list[:, h][int(targets[h])] = 1

        out_grad = - (label_list - out_softmax)
        h2_a_grad = np.transpose(self.weights_h2_o).dot(out_grad)
        h2_w_grad = out_grad.dot(np.transpose(arr2))
        h2_b_grad = np.array([np.sum(out_grad[i]) for i in range(out_grad.shape[0])])
        # print(out_grad.shape)
        # print(h2_a_grad.shape)
        # print(h2_w_grad.shape)
        # print(h2_b_grad.shape)
        # print('--------')

        h2_grad = np.multiply(h2_a_grad, np.array([sig_derivative(z2[p]) for p in range(z2.shape[0])]))
        h1_a_grad = np.transpose(self.weights_h1_h2).dot(h2_grad)
        h1_w_grad = h2_grad.dot(np.transpose(arr))
        h1_b_grad = np.array([np.sum(h2_grad[i]) for i in range(h2_grad.shape[0])])
        # print(h2_grad.shape)
        # print(h1_a_grad.shape)
        # print(h1_w_grad.shape)
        # print(h1_b_grad.shape)
        # print('--------')

        h1_grad = np.multiply(h1_a_grad, np.array([sig_derivative(z1[p]) for p in range(z1.shape[0])]))
        # in_a_grad = np.dot(h1_grad, np.transpose(self.weights_i_h1))
        in_w_grad = h1_grad.dot(np.transpose(inputs))
        in_b_grad = np.array([np.sum(h1_grad[i]) for i in range(h1_grad.shape[0])])
        # print(h1_grad.shape)
        # print(in_w_grad.shape)
        # print(in_b_grad.shape)
        # print('--------')

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
            end = int(size[0] / 10)
            for _ in range(10):
                batches.append(np.array(dataset[start:end]))
                start = end
                end = end + int(size[0] / 10)
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
    # test_labels = read(sys.argv[4])

    epochs = 100
    my_agent = NeuralNetwork(784, 256, 256, 10)
    learning_rate = 0.003

    train_batches = get_batches(train_inputs)
    label_batches = get_batches(train_labels)

    for e in range(epochs):
        for g in range(len(train_batches)):
            a2, b2, c2, d2, e2, f2 = my_agent.feed_f(train_batches[g].T)
            temp_h2_w, temp_h2_b, temp_h1_w, temp_h1_b, temp_in_w, temp_in_b = my_agent.back_p(a2, label_batches[g], b2, c2, d2, e2, f2)
            h2_out_bias = np.reshape(temp_h2_b, (my_agent.out, 1))
            h1_h2_bias = np.reshape(temp_h1_b, (my_agent.h2, 1))
            in_h1_bias = np.reshape(temp_in_b, (my_agent.h1, 1))

            my_agent.weights_i_h1 = my_agent.weights_i_h1 - (learning_rate * (temp_in_w / train_batches[g].shape[0]))
            my_agent.weights_h1_h2 = my_agent.weights_h1_h2 - (learning_rate * (temp_h1_w / train_batches[g].shape[0]))
            my_agent.weights_h2_o = my_agent.weights_h2_o - (learning_rate * (temp_h2_w / train_batches[g].shape[0]))

            my_agent.h2_bias = my_agent.h2_bias - (learning_rate * (h1_h2_bias / train_batches[g].shape[0]))
            my_agent.h1_bias = my_agent.h1_bias - (learning_rate * (in_h1_bias / train_batches[g].shape[0]))
            my_agent.out_bias = my_agent.out_bias - (learning_rate * (h2_out_bias / train_batches[g].shape[0]))

    # correct = 0
    a1, c1, d1, e1, f1, g1 = my_agent.feed_f(test_inputs.T)
    list_1 = np.argmax(a1, axis=0)
    list_1 = np.reshape(list_1, (test_inputs.shape[0], 1))
    # test_l = np.reshape(test_labels.astype(int), (test_labels.shape[0], 1))
    # for n in range(test_l.shape[0]):
    #     if list_1[n] == test_l[n]:
    #         correct = correct + 1
    #
    # print(correct)
    # print(test_labels.shape[0])
    # print(100 * (correct / test_labels.shape[0]))

    # answer = np.array(list_1)
    pd.DataFrame(list_1).to_csv('test_predictions.csv', header=False, index=False)

    print('done')


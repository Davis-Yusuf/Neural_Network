import numpy as np
import math
import sys


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
        self.h2_bias = np.ones(h2)
        self.h1_bias = np.ones(h1)
        self.out_bias = np.ones(out)
        for _ in range(h1 - 1):
            self.weights_i_h1 = np.vstack((self.weights_i_h1, np.random.uniform(-1, 1, inputs)))
        for _ in range(h2 - 1):
            self.weights_h1_h2 = np.vstack((self.weights_h1_h2, np.random.uniform(-1, 1, h2)))
        for _ in range(out - 1):
            self.weights_h2_o = np.vstack((self.weights_h2_o, np.random.uniform(-1, 1, h2)))

    def feed_f(self, inputs, targets):

        z1_output = np.sum(np.matmul(inputs / 255, self.weights_i_h1[0])) + self.h1_bias[0]
        h1_output = sig_activation_function(z1_output)
        label_list = np.zeros(self.out)

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
            label_list[int(targets)] = 1
            # out_output[k] = softmax_activation_function(out_output[k])

        final = softmax_activation_function(out_output)
        # temp = cross_entropy_func(label_list, final
        new_labels = label_list.reshape((10, 1))

        return final, new_labels, z1_output, z2_output, h1_output, h2_output, inputs

    def back_p(self, out_softmax, labels, z1, z2, arr, arr2, inputs, learning_rate):
        costs = {}

        out_grad = np.subtract(out_softmax, labels)
        h2_a_grad = np.dot(np.transpose(self.weights_h2_o), out_grad)
        h2_w_grad = learning_rate * np.dot(out_grad, np.transpose(arr2))
        h2_b_grad = out_grad
        self.weights_h2_o = self.weights_h2_o + h2_w_grad
        self.out_bias = self.out_bias + h2_b_grad.flatten()

        h2_grad = np.multiply(h2_a_grad, sig_derivative(z2))
        h1_a_grad = np.dot(np.transpose(self.weights_h1_h2), h2_grad)
        h1_w_grad = learning_rate * np.dot(h2_grad, np.transpose(arr))
        h1_b_grad = h2_grad
        self.weights_h1_h2 = self.weights_h1_h2 + h1_w_grad
        self.h2_bias = self.h2_bias + h1_b_grad.flatten()

        h1_grad = np.multiply(h1_a_grad, sig_derivative(z1))
        # in_a_grad = np.dot(h1_grad, np.transpose(self.weights_i_h1))
        new_inputs = inputs.reshape((784, 1))
        in_w_grad = learning_rate * np.dot(h1_grad, np.transpose(new_inputs))
        in_b_grad = h1_grad
        self.weights_i_h1 = self.weights_i_h1 + in_w_grad
        self.h1_bias = self.h1_bias + in_b_grad.flatten()


def read(file_name):
    with open(file_name) as file:
        dataset = np.genfromtxt(file, delimiter=",")

    return dataset


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

    return batches


if __name__ == "__main__":
    train_inputs = read(sys.argv[1])
    train_labels = read(sys.argv[2])

    training = train_inputs[:5200]
    testing = train_inputs[5200:]
    training_l = train_labels[:5200]
    testing_l = train_labels[5200:]

    total = testing.shape[0]
    correct = 0
    my_agent = NeuralNetwork(784, 20, 20, 10)

    print(training.shape)
    print(training_l.shape)
    print(testing.shape)
    print(testing_l.shape)

    for m in range(1000):
        a, b, c, d, e, f, g = my_agent.feed_f(training[m], training_l[m])
        my_agent.back_p(a, b, c, d, e, f, g, 0.001)

    for n in range(testing.shape[0]):
        a, b, c, d, e, f, g = my_agent.feed_f(testing[n], testing_l[n])
        print(np.amax(a), a)
        answer = np.where(a == np.amax(a))
        print(answer)
        print(answer[0], testing_l[n])
        quit()
        if answer == testing_l[n]:
            correct = correct + 1
    print(100 * (correct/total))

    print('done')

    # all_batches = get_batches(train_inputs)
    #
    # for i in range(len(all_batches)):
    #     temp_batches = all_batches[i]
    #     for j in range(all_batches[0].shape[0]):
    #         input_nodes = temp_batches[j] / 255
    #         # my_agent.inputs = input_nodes
    #         # my_agent.feed_f()


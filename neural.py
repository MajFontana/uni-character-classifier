import numpy
import math
import random



class Sigmoid:

    def __call__(self, x):
        return 1 / (1 + numpy.e ** (-x))

    def diff(self, y):
        return y * (1 - y)

class HalfMeanSquaredError:

    def __call__(self, output, target):
        return (target - output) ** 2 / 2

    def diff(self, output, target):
        return output - target



class Argmax:

    def __init__(self, size):
        self.vectors = numpy.hsplit(numpy.eye(size), size)

    def __call__(self, vector):
        return numpy.argmax(vector)

    def vector(self, index):
        return self.vectors[index]

class Dense:

    def __init__(self, input_size, output_size, activation_class=Sigmoid):
        self.weights = numpy.random.uniform(0, 1, (output_size, input_size))
        self.biases = numpy.zeros((output_size, 1))
        self.activation = activation_class()

    def __call__(self, input):
        return self.activation(numpy.matmul(self.weights, input)) + self.biases

    def delta(self, output, next_delta):
        return self.activation.diff(output) * next_delta

class Sequential:

    def __init__(self, input_size, architecture, loss_class=HalfMeanSquaredError):
        self.layers = []
        for layer_class, output_size in architecture:
            self.layers.append(layer_class(input_size, output_size))
            input_size = output_size
        self.loss = loss_class()

    def __call__(self, input):
        vector = input
        for layer in self.layers:
            vector = layer(vector)
        return vector

    def training_step(self, input, target):
        # Forward pass
        inputs = []
        inputs.append(input)
        for layer in self.layers:
            inputs.append(layer(inputs[-1]))

        # Backward pass
        deltas = [self.loss.diff(inputs[-1], target)]
        for l in range(len(self.layers) - 2, -1, -1):
            next_delta = numpy.matmul(self.layers[l + 1].weights.T, deltas[0])
            deltas.insert(0, self.layers[l].delta(inputs[l + 1], next_delta))

        gradients = [numpy.matmul(deltas[l], inputs[l].T) for l in range(len(self.layers))]
        return gradients
    
    def fit(self, input_batch, target_batch, epochs, learning_rate):
        for epoch in range(epochs):
            gradients = [numpy.zeros(layer.weights.shape) for layer in self.layers]
            for batch_index in range(len(input_batch)):
                input = input_batch[batch_index]
                target = target_batch[batch_index]
                stochastic_gradients = self.training_step(input, target)
                for l in range(len(gradients)):
                    gradients[l] += stochastic_gradients[l]
            for l in range(len(gradients)):
                gradients[l] /= len(gradients)
            for l in range(len(gradients)):
                self.layers[l].weights -= gradients[l] * learning_rate



class Classifier:

    def __init__(self, input_size, architecture, labels, output_layer_class=Dense):
        self.input_size = input_size
        indices = list(range(len(labels)))
        self.label_dictionary = {label: index for index, label in zip(indices, labels)}
        self.index_dictionary = {index: label for index, label in zip(indices, labels)}
        
        self.samples = [[] for _ in range(len(indices))]
        self.argmax = Argmax(len(indices))
        new_architecture = architecture + [(output_layer_class, len(indices))]
        self.model = Sequential(input_size, new_architecture)

    def add_samples(self, inputs, labels):
        for input, label in zip(inputs, labels):
            index = self.label_dictionary[label]
            self.samples[index].append(input)

    def train(self, epochs, learning_rate, sample_size=None):
        if sample_size == None:
            # all samples will be trained on
            sample_size = max([len(inputs) for inputs in self.samples])
        
        # create a training batch
        input_batch = []
        target_batch = []
        indices = list(self.index_dictionary.keys())
        for sample_index in range(sample_size):
            for index in random.sample(indices, len(indices)):
                inputs = self.samples[index]
                input = inputs[sample_index % len(inputs)]
                input_batch.append(input)
                target = self.argmax.vector(index)
                target_batch.append(target)

        # fit model
        self.model.fit(input_batch, target_batch, epochs, learning_rate)

    def validate(self):
        # create a validation batch
        input_batch = []
        index_batch = []
        indices = self.index_dictionary.keys()
        for index in indices:
            inputs = self.samples[index]
            input_batch += inputs
            index_batch += [index] * len(inputs)

        # predict outputs
        output_batch = self.model.predict(input_batch)

        # calculate accuracy
        result = [self.argmax(output) == index for output, index in zip(output_batch, index_batch)]
        accuracy = result.count(True) / len(result)
        
        return accuracy

    def classify(self, vector):
        return self.index_dictionary[self.argmax(self.model(vector))]

from numpy import exp, random, dot
import numpy as np

class NetworkLayer():
    # Creates a single layer for a fully connected NN.
    def __init__(self, num_neurons, use_bias = False, bias_value = 1.0):
        self.use_bias = use_bias
        self.bias_value = bias_value

        # The bias is neuron who's output is always the weight.
        # Acts as the 'b' in y = ax + b to shift the activation function
        # to better fit the prediction
        self.num_neurons = num_neurons + use_bias
        self.weights = None
        self.input_count = None
        self.in_network = False

    def generate_weights(self, input_count):
        self.in_network = True
        self.input_count = input_count
        # Creates random weights between (-1, 1)
        self.weights = np.array(2 * random.random((input_count, self.num_neurons)) - 1)

class LayerStack():
    # Initializes the layer stick with the first layer
    # that knows the input size
    def __init__(self):
        self.layers = []

    def set_first_layer(self, input_size):
        if len(self.layers) == 0:
            print("Can't set input dims with no input layer.")
            raise SystemExit(0)

        self.layers[0].generate_weights(input_size)

    def length(self):
        return len(self.layers)

    # Appends another layer and creates (# inputs * # num_neuron) weights
    def append(self, layer):
        if (self.length() > 0):
            prev_layer = self.layers[self.length() - 1]
            layer_inputs = prev_layer.num_neurons
            layer.generate_weights(layer_inputs)
            self.layers.append(layer)
        else:
            self.layers.append(layer)

class FCNN():
    def __init__(self, layer_stack, seed = 253):
        self.net = layer_stack

    def sig_deriv(self, x):
        return x * (1 - x)

    def sig(self, x):
        return 1 / (1 + exp(-x))

    def infer(self, inputs):
        output = []
        for x in range(0, self.layers.length):
            current_output = None
            current_layer = self.layers[x]
            
            if x == 0:
                prev_out = inputs
            else:
                prev_out = np.copy(output[x-1])

            current_output = self.sig(dot(prev_out, current_layer))
            if (current_layer.use_bias is True):
                current_output[::, current_layer.weights.shape[1] - 1] = current_layer.bias_value

            output.append(current_output)

        return output

    def train(self, training_data, expected_outputs, learning_rate = 0.1, momentum = 0.0, epsilon = 1e-10, iterations = 50000):
        self.net.set_first_layer(training_data.shape[1])
        num_layers = self.net.length()
        prev1_weight = [0 for i in range(num_layers)]
        prev2_weight = [0 for i in range(num_layers)]
        momentum_shift = 0

        for iteration in range(iterations):
            layer_outputs = self.infer(training_data)
            layer_error = [0 for i in range(num_layers)]
            layer_delta = [0 for i in range(num_layers)]
            layer_weight_shift = [0 for i in range(num_layers)]

            # Back-propogates from the output layer to the input layers
            for i in range(num_layers - 1, -1, -1):

                # If it's the output layer, subtract the expected output by the output.
                # If expected_output = layer_output, the result will be 0 indicating no error.
                if i == num_layers - 1:
                    layer_error[i] = expected_outputs - layer_outputs[i]
                # Otherwise dot product the following layer's delta with its weights.
                else:
                    layer_error[i] = layer_delta[i+1].dot(self.net.layers[i+1].weights.T)

                # Delta is calculated by multiplying the error with the sigmoid derivative of
                # the current layer's output.
                layer_delta[i] = layer_error[i] * self.sig_deriv(layer_outputs[i])

                # If it's the first layer, dot the layer delta with the training data                
                if i == 0:
                    layer_weight_shift[i] = training_data.T.dot(layer_delta[i])
                # Otherwise dot with the previous layer's output
                else:
                    layer_weight_shift[i] = layer_outputs[i-1].T.dot(layer_delta[i])
            
                print("Layer outputs:\n")
                print(layer_outputs[i])
                print("\n")
                print("Layer Error:\n")
                print(layer_error[i])
                print("\n")
                print("Layer Delta:\n")
                print(layer_delta[i])
                print("\n")
                print("Layer weight shift\n")
                print(layer_weight_shift[i])
                print("\n")

            for i in range(0, num_layers):
                if (iteration == 0):
                    self.net.layers[i].weights += (learning_rate * layer_weight_shift[i])
                elif (iteration == 1):
                    prev1_weight[i] = np.copy(self.net.layers[i].weights)
                    self.net.layers[i].weights += (learning_rate * layer_weight_shift[i])
                else:
                    prev2_weight[i] = np.copy(prev1_weight[i])
                    prev1_weight[i] = np.copy(self.net.layers[i].weights)
                    momentum_shift = momentum * (prev1_weight[i] - prev2_weight[i])
                    self.net.layers[i].weights += ((1 - momentum) * learning_rate * layer_weight_shift[i]) + momentum_shift

            if all(abs(layer_error[num_layers-1])) < epsilon:
                print("Training convergence at " + str(iteration + 1) + " iterations.\n")
                break
        print("Training did not converge. Ran " + str(iteration + 1) + " iterations.\n")
            

    def infer(self, inputs):
        output = []

        for i in range(0, self.net.length()):
            prev_output = None
            current_output = None
            current_layer = self.net.layers[i]

            if i == 0:
                prev_output = inputs
            else:
                prev_output = np.copy(output[i-1])

            current_output = self.sig(prev_output.dot(current_layer.weights))

            if (current_layer.use_bias is True):
                if (len(current_output.shape) > 1):
                    current_output[::, current_layer.weights.shape[1] - 1] = current_layer.bias_value
                else:
                    current_output[current_layer.weights.shape[1] - 1] = current_layer.bias_value

            output.append(current_output)

        return output


def main():
    layer_stack = LayerStack()
    lyr_0 = NetworkLayer(4, True)
    lyr_1 = NetworkLayer(2, True)
    lyr_2 = NetworkLayer(8)
    lyr_3 = NetworkLayer(5)
    lyr_4 = NetworkLayer(1)
    lyr_5 = NetworkLayer(5)
    lyr_6 = NetworkLayer(1, True)

    layer_stack.append(lyr_0)
    layer_stack.append(lyr_6)

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1, 1, 0]]).T

    neural_net = FCNN(layer_stack)
    neural_net.train(x, y, learning_rate=1.0, momentum=0.0, epsilon=1e-1, iterations=1)

    x0_test = np.array([0, 1, 0])
    result = neural_net.infer(x0_test)[neural_net.net.length() - 1]
    print("Result should be 1")
    print(result)

    x1_test = np.array([1, 1, 1])
    result = neural_net.infer(x1_test)[neural_net.net.length() - 1]
    print("Result should be 0")
    print(result)
main()

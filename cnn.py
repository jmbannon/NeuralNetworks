import numpy as np

class ConvolutionalND(object):
    def __init__(self, n, input_shape, filters, filter_shape, stride=1, padding=0, activation=None):
        assert isinstance(n, int) and n > 0
        assert ParamCheck.is_shape_tuple(input_shape, n)
        assert ParamCheck.is_shape_tuple(filter_shape, n)

        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.

    @classmethod
    def _output_shape(n, input_shape, filter_shape, padding, stride):
        

class Convolution3D(ConvolutionalND):

    ########################################################################
    # @brief Initializes a 3D convolutional layer
    # @param input_shape 3-tuple of input shape (nr_rows, nr_cols, nr_channels)
    # @param filters Integer specifying nr_filters or array of filters of equal shape
    # @param filter_shape 3-tuple of filter shape (nr_rows, nr_cols, nr_channels). nr_channels should match input nr_channels
    # @param stride Integer defining stride of filter on input
    # @param padding Integer defining padding amount on input
    # @param activation Activation function applied after filtering
    # @param weights Weights definable by user
    # @param bis Bias initialized by user
    # @param bias_flag Boolean; whether to user bias or not
    #
    def __init__(self, input_shape, filters, filter_shape, stride=1, padding=0, activation=None, weights, bias, bias_flag=True):
        assert ParamCheck.is_shape_tuple(input_shape, 3)
        assert ParamCheck.is_shape_tuple(filter_shape, 3)

        # TODO init filters, weights, etc
        self.input_shape = input_shape
        self.output_shape = Convolution3D._output_shape(input_shape, filter_shape, padding, stride)

    #########################################################################
    # @brief Output shape equation based on input shape, filter shape, padding, and stride:
    #   (W - F + 2P) / S + 1 where
    #       W = input_shape
    #       F = filter_shape
    #       P = padding
    #       S = stride
    @classmethod
    def _output_shape(input_shape, filter_shape, padding, stride):
        padding_increase = (2 * padding, 2 * padding, 0)
        stride_div = (stride + 1, stride + 1, 0)
        return (input_shape - filter_shape + padding_increase) / stride_div

    def pad_input(input_layer):
        axis_increase = self.padding * 2
        result_shape = input.shape + (axis_increase, axis_increase, 0)
        result = np.zeros(result_shape)
        result[self.padding:input.shape[0], self.padding:input.shape[1], :] = input_layer
        return result

    def convolution(input_layer):
        if padding > 0:
            input_layer = pad_input(input_layer)


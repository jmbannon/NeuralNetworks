import numpy as np

class ParamCheck(object):
    
    @staticmethod
    def is_shape_tuple(shape, nr_dims):
        return isinstance(shape, tuple) and len(shape) == nr_dims and all(isinstance(i, int) for i in shape)

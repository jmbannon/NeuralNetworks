package Layers

import NNMath.Volume

/**
  * Created by jb on 9/26/16.
  */
class Convolutional2D(input_shape: Shape,
                      output_shape: Shape,
                      nb_filter: Int,
                      nb_shape: Shape,
                      nb_stride: (Int, Int),
                      weights: Volume,
                      bias: Volume,
                      _name: String) extends Layer(input_shape, output_shape, _name)
{

}

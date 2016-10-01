package Layers

import scala.annotation.meta.{param, setter}

/**
  * Created by jb on 9/30/16.
  */
abstract class Layer(var _input_shape: Int = -1,
                     var _output_shape: Int = -1,
                     var _name : String = null) {

  private def valid_shape(shape: Int) : Boolean = {
    return shape > 0
  }

  require(valid_shape(_input_shape),  "output_shape must be a positive Int or Tuple of Ints")
  require(valid_shape(_output_shape),  "output_shape must be a positive Int or Tuple of Ints")

  @param def input_shape : Int = { _input_shape }
  @param def output_shape : Int = { _output_shape }

  @setter def input_shape_= (that: Int) { _input_shape = that }
  @setter def output_shape_= (that: Int) { _output_shape = that }

  def display_name(layer_idx: Int): String = {
    val stb = StringBuilder.newBuilder
    stb.append("Layer ")
    stb.append(layer_idx)

    if (_name != null) {
      stb.append(" [")
      stb.append(_name)
      stb.append("]")
    }
    stb.append(":")
    return stb.toString()
  }

}

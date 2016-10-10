package Layers

import scala.annotation.meta.{param, setter}

/**
  * Created by jb on 9/30/16.
  */
abstract class Layer(var _input_shape: Shape = null,
                     var _output_shape: Shape = null,
                     var _name : String = null) {

  @param def input_shape : Shape = _input_shape
  @param def output_shape : Shape = _output_shape

  @setter def input_shape_= (that: Shape) { _input_shape = that }
  @setter def output_shape_= (that: Shape) { _output_shape = that }

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

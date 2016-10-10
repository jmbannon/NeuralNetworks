package Layers

/**
  * Created by jb on 10/7/16.
  */
object Shape {
  private val ERROR_UNDEFINED_SHAPE = "Undefined Shape"
  private val ERROR_OUT_OF_BOUNDS = "Attempted to get dimension that is out of bounds"

  val UNDEFINED_DIMENSION = -253
  val UNDEFINED_SHAPE = new Shape(UNDEFINED_DIMENSION, UNDEFINED_DIMENSION, UNDEFINED_DIMENSION)
}

class Shape(val shape: (Int, Int, Int)) {

  val _is1D = shape._1 > 0 && shape._2 == Shape.UNDEFINED_DIMENSION && shape._3 == Shape.UNDEFINED_DIMENSION
  val _is2D = shape._1 > 0 && shape._2 > 0 && shape._3 == Shape.UNDEFINED_DIMENSION
  val _is3D = shape._1 > 0 && shape._2 > 0 && shape._3 > 0
  val _isUndefined = shape._1 == Shape.UNDEFINED_DIMENSION && shape._2 == Shape.UNDEFINED_DIMENSION && shape._3 == Shape.UNDEFINED_DIMENSION

  require(_is1D || _is2D || _is3D || _isUndefined, "Invalid shape. Must be 1-3 dimensional with positive integers.")

  def this(shape: Int)         { this((shape, Shape.UNDEFINED_DIMENSION, Shape.UNDEFINED_DIMENSION)) }
  def this(shape: Tuple1[Int]) { this((shape._1, Shape.UNDEFINED_DIMENSION, Shape.UNDEFINED_DIMENSION)) }
  def this(shape: (Int, Int))  { this((shape._1, shape._2, Shape.UNDEFINED_DIMENSION)) }

  def is1D: Boolean = _is1D
  def is2D: Boolean = _is2D
  def is3D: Boolean = _is3D
  def isDefined: Boolean = !_isUndefined
  def isUndefined: Boolean = _isUndefined

  def x: Int = { assert(isDefined, Shape.ERROR_UNDEFINED_SHAPE); shape._1 }
  def y: Int = { assert(isDefined, Shape.ERROR_UNDEFINED_SHAPE); assert(!_is1D, Shape.ERROR_OUT_OF_BOUNDS); shape._2 }
  def z: Int = { assert(isDefined, Shape.ERROR_UNDEFINED_SHAPE); assert(_is3D, Shape.ERROR_OUT_OF_BOUNDS); shape._3 }

  def apply(nDimension: Int): Int = {
    nDimension match {
      case 0 => return this.x
      case 1 => return this.y
      case 2 => return this.z
    }

    assert(true, Shape.ERROR_OUT_OF_BOUNDS)
    return Shape.UNDEFINED_DIMENSION
  }

  private def wrapParenthesis(str: String) = { '(' + str + ')' }

  override def toString: String = {
    val stb = new StringBuilder
    if (_is1D) {
      stb.append(shape._1)
    } else if (_is2D) {
      stb.append(shape._1)
      stb.append(", ")
      stb.append(shape._2)
    } else if (_is3D) {
      stb.append(shape._1)
      stb.append(", ")
      stb.append(shape._2)
      stb.append(", ")
      stb.append(shape._3)
    } else {
      stb.append("(Undefined)")
    }
    return stb.toString()
  }
}

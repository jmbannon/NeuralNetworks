package NNMath

import Layers.Shape
import breeze.linalg.DenseMatrix

/**
  * Created by jb on 10/10/16.
  */
object Volume {

  /**
    * @param shape Shape of the zero-matrices.
    * @return Array of zero-matrices of size shape.
    */
  def zeroArray(shape: Shape): Array[DenseMatrix[Double]] = {
    assert(shape.is3D, "Shape must be 3-D")
    val vol = new Array[DenseMatrix[Double]](shape.z)
    for (i <- vol.indices) {
      vol(i) = DenseMatrix.zeros(shape.x, shape.y)
    }
    vol
  }

  def zeroArray(x: Int, y: Int, z: Int): Array[DenseMatrix[Double]] = { zeroArray(new Shape(x, y, z)) }
  def zero(shape: Shape): Volume = { new Volume(zeroArray(shape)) }
  def zero(x: Int, y: Int, z: Int): Volume = { zero(new Shape(x, y, z)) }

  def randArray(shape: Shape): Array[DenseMatrix[Double]] = {
    assert(shape.is3D, "Shape must be 3-D")
    val vol = new Array[DenseMatrix[Double]](shape.z)
    for (i <- vol.indices) {
      vol(i) = DenseMatrix.rand(shape.x, shape.y)
    }
    vol
  }

  def randArray(x: Int, y: Int, z: Int): Array[DenseMatrix[Double]] = { randArray(new Shape(x, y, z)) }
  def rand(shape: Shape): Volume = { new Volume(randArray(shape)) }
  def rand(x: Int, y: Int, z: Int): Volume = { new Volume(randArray(x, y, z)) }

  /**
    * @param shape Shape of the matrices.
    * @return Array of matrices of size shape.
    */
  def array(shape: Shape): Array[DenseMatrix[Double]] = {
    assert(shape.is3D, "Shape must be 3-D")
    val vol = new Array[DenseMatrix[Double]](shape.z)
    for (i <- vol.indices) {
      vol(i) = new DenseMatrix(shape.x, shape.y)
    }
    vol
  }

  def array(x: Int, y: Int, z: Int): Array[DenseMatrix[Double]] = { array(new Shape(x, y, z)) }
  def apply(shape: Shape): Volume = { new Volume(array(shape)) }
  def apply(x: Int, y: Int, z: Int): Volume = { Volume(new Shape(x, y, z)) }

  /**
    * Subsets a volume using range of x, y, and z.
    *
    * @param vol Volume to subset.
    * @param xRange X subset range.
    * @param yRange Y subset range.
    * @param zRange Z subset range.
    * @return Subset of Volume object.
    */
  def sliceVolumeZ(vol: Volume, xRange: Range, yRange: Range, zRange: Range): Volume = {
    val volArray = array(new Shape(xRange.length, yRange.length, zRange.length))
    for (i <- zRange.zipWithIndex) {
      volArray(i._2) = vol.layers(i._1)(xRange, yRange)
    }
    new Volume(volArray)
  }

  /**
    * Subsets a volume using range of x, y. Subsets all layers (z).
    *
    * @param vol Volume to subset.
    * @param xRange X subset range.
    * @param yRange Y subset range.
    * @return Subset of Volume object.
    */
  def sliceVolume(vol: Volume, xRange: Range, yRange: Range): Volume = {
    sliceVolumeZ(vol, xRange, yRange, vol.layers.indices)
  }

  def pad(vol: Volume, p: Int): Volume = {
    val toRet = new Volume(new Array[DenseMatrix[Double]](vol.depth))
    for (i <- toRet.indices) {
      toRet(i) := DenseMatrix.zeros[Double](vol.nrows + p, vol.ncols + p)
      toRet(i)(p to vol.nrows - p, p to vol.ncols - p) := vol.layers(i)
    }
    return toRet
  }
}

/**
  * Represents a 3-dimension matrix used in Convolutional Neural Networks.
  *
  * @param layers Array of matrices representing volume. Must all be of the same dimensions.
  */
class Volume(var layers: Array[DenseMatrix[Double]]) {
  require(layers.length >= 1, "Volume array of matrices must be >= 1")
  val nrows = layers(0).rows
  val ncols = layers(0).cols
  val depth = layers.length

  for (i <- layers.indices) {
    require(layers(i).rows == nrows, "row mismatch in volume: " + layers(i).rows + " != " + nrows)
    require(layers(i).cols == ncols, "col mismatch in volume: " + layers(i).cols + " != " + ncols)
  }

  val shape = new Shape(nrows, ncols, depth)

  def this(shape: Shape) { this(Volume.zeroArray(shape)) }

  def apply(layer: Int): DenseMatrix[Double] = { layers(layer) }
  def apply(xRange: Range, yRange: Range, zRange: Range): Volume = { Volume.sliceVolumeZ(this, xRange, yRange, zRange) }
  def apply(xRange: Range, yRange: Range): Volume = { Volume.sliceVolume(this, xRange, yRange) }
  def indices: Range = { layers.indices }

  def pad(p: Int): Volume = { Volume.pad(this, p) }

  override def toString(): String = {
    val stb = new StringBuilder
    stb.append("Volume of shape: ")
    stb.append(shape.toString)
    stb.append('\n')
    for (i <- layers.indices) {
      stb.append("Layer " + i + ":\n")
      stb.append(layers(i).toString())
      stb.append("\n\n")
    }
    stb.toString()
  }
}

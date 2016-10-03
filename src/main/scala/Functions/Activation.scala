package Functions

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by jb on 9/30/16.
  */
abstract class Activation(val _name : String) {
  def name = _name
  def apply(input: Double) : Double
  def d(input: Double) : Double

  def apply(input: DenseVector[Double]) : DenseVector[Double]
  def d(input: DenseVector[Double]) : DenseVector[Double]

  def apply(input: DenseMatrix[Double]) : DenseMatrix[Double] = {
    for (i <- 0 to input.rows - 1) {
      input(i, ::) := apply(input(i, ::).t).t
    }
    input
  }

  def d(input: DenseMatrix[Double]) : DenseMatrix[Double] = {
    for (i <- 0 to input.rows - 1) {
      input(i, ::) := d(input(i, ::).t).t
    }
    input
  }
}

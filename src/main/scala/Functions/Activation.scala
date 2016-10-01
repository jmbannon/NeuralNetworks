package Functions

import breeze.linalg.DenseVector

/**
  * Created by jb on 9/30/16.
  */
abstract class Activation(val _name : String) {
  def name = _name
  def apply(input: Double) : Double
  def d(input: Double) : Double

  def apply(input: DenseVector[Double]) : DenseVector[Double]
  def d(input: DenseVector[Double]) : DenseVector[Double]
}

package Functions

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by jb on 9/30/16.
  */
abstract class Cost(val _name: String) {
  def name = _name
  def apply(output: Double, exp_output: Double) : Double
  def d(input: Double, exp_output: Double) : Double

  def apply(output: DenseVector[Double], exp_output: DenseVector[Double]) : Double
  def d(input: DenseVector[Double], exp_output: DenseVector[Double]) : DenseVector[Double]
}

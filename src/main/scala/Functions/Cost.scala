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

  def apply(output: DenseMatrix[Double], exp_output: DenseMatrix[Double]) : DenseVector[Double] = {
    val to_ret : DenseVector[Double] = DenseVector.zeros(output.size)
    for (i <- 0 to (output.rows - 1)) {
      to_ret(i) = apply(output(i, ::).t, exp_output(i, ::).t)
    }
    to_ret
  }

  def d(output: DenseMatrix[Double], exp_output: DenseMatrix[Double]) : DenseMatrix[Double] = {
    for (i <- 0 to (output.rows - 1)) {
      output(i, ::) := d(output(i, ::).t, exp_output(i, ::).t).t
    }
    output
  }

}

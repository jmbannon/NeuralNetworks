package Functions

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by jb on 9/30/16.
  */
abstract class Activation(val _name : String) {
  def name = _name
  def apply(x: Double) : Double
  def d(x: Double) : Double

  def apply(x: DenseVector[Double]) : DenseVector[Double] = { x.map(xi => apply(xi)) }
  def d(x: DenseVector[Double]) : DenseVector[Double] = { x.map(xi => d(xi)) }

  def apply(x: DenseMatrix[Double]) : DenseMatrix[Double] = { x.map(xi => apply(xi)) }
  def d(x: DenseMatrix[Double]) : DenseMatrix[Double] = { x.map(xi => d(xi)) }
}

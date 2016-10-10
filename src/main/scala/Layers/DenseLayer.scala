package Layers

import Functions.{Activation, Activations}
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.abs
import breeze.stats.distributions.Rand

import scala.annotation.meta.{getter, param, setter}

/**
  * Created by jb on 9/26/16.
  */

object DenseLayer {
  def apply(output_dim: Int,
            input_dim: Int = Shape.UNDEFINED_DIMENSION,
            activation: String = "linear",
            weights: DenseMatrix[Double] = null,
            bias: DenseVector[Double] = null,
            use_bias: Boolean = true,
            name: String = null): DenseLayer = {
    new DenseLayer(new Shape(output_dim), new Shape(input_dim), activation, weights, bias, use_bias, name)
  }
}

class DenseLayer(output_shape: Shape,
                 input_shape: Shape,
                 var activation_str: String,
                 var weights: DenseMatrix[Double],
                 var bias: DenseVector[Double],
                 var use_bias: Boolean,
                 _name: String) extends Layer(input_shape, output_shape, _name) {

  activation_str = activation_str.toLowerCase
  require(Activations.exists(activation_str), "Activation function " + activation_str + " does not exist")

  val activation: Activation = Activations(activation_str)

  def updateWeights(weightShift: DenseMatrix[Double]) = { weights += weightShift }
  def updateBias(biasShift: DenseVector[Double]) = { biasShift += biasShift }

  def contains_weights : Boolean = { weights != null && (if (use_bias) bias != null else true) }
  def gen_weights = {
    if (weights == null) {
      weights = DenseMatrix.rand(super.input_shape(0), super.output_shape(0))
    }
    if (use_bias && bias == null) {
      bias = DenseVector.rand(super.output_shape(0))
    }
  }

  def gradient(inputs: DenseVector[Double]) : DenseVector[Double] = activation.d(inputs)
  def gradient(inputs: DenseMatrix[Double]) : DenseMatrix[Double] = activation.d(inputs)

  /** Predict with a single sample of training data. */
  def step(inputs : DenseVector[Double]) : DenseVector[Double] = {
    val cell_body = (inputs.asDenseMatrix * weights).toDenseVector

    if (use_bias) cell_body += bias
    return activation(cell_body)
  }

  /** Predict with multiple samples of training data. */
  def step(inputs: DenseMatrix[Double]) : DenseMatrix[Double] = {
    var cell_body = inputs * weights
    if (use_bias) cell_body = cell_body(*, ::).map(row => row + bias)
    return activation(cell_body)
  }
}

package Layers

import Functions.{Activation, Activations}
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.abs
import breeze.stats.distributions.Rand

import scala.annotation.meta.{getter, param, setter}

/**
  * Created by jb on 9/26/16.
  */
class DenseLayer(_output_dim: Int,
                 _input_dim: Int = -1,
                 var activation: String = "linear",
                 var _weights: DenseMatrix[Double] = null,
                 var _bias: DenseVector[Double] = null,
                 var _use_bias: Boolean = true,
                 _name: String = null) extends Layer(_input_dim, _output_dim, _name) {

  activation = activation.toLowerCase
  require(Activations.exists(activation), "Activation function " + activation + " does not exist")
  //require(_bias.length == _output_dim, "Bias vector must be of length _output_dim")

  private val _activation : Activation = Activations(activation)

  @param def step : Activation = { _activation }

  @getter def bias = _bias
  @getter def weights = _weights

  @setter def bias_= (that: DenseVector[Double]) { _bias = that }
  @setter def weights_= (that: DenseMatrix[Double]) { _weights = that }

  def has_weights : Boolean = { _weights != null && (if (_use_bias) _bias != null else true) }
  def gen_weights = {
    if (_weights == null) {
      _weights = abs(DenseMatrix.rand(super.input_shape, super.output_shape))
    }
    if (_use_bias && _bias == null) {
      _bias = abs(DenseVector.rand(super.output_shape))
    }
  }

  def gradient(inputs: DenseVector[Double]) : DenseVector[Double] = _activation.d(inputs)
  def gradient(inputs: DenseMatrix[Double]) : DenseMatrix[Double] = _activation.d(inputs)

  /** Predict with a single sample of training data. */
  def predict(inputs : DenseVector[Double]) : DenseVector[Double] = {
    val cell_body : DenseVector[Double] = (inputs.asDenseMatrix * _weights).toDenseVector

    if (_use_bias) cell_body += _bias
    return _activation(cell_body)
  }

  /** Predict with multiple samples of training data. */
  def predict(inputs : DenseMatrix[Double]) : DenseMatrix[Double] = {
    var cell_body : DenseMatrix[Double] = inputs * _weights
    if (_use_bias) cell_body = cell_body(*, ::).map(row => row + _bias)
    return _activation(cell_body)
  }
}

package Layers

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}

/**
  * Created by jb on 9/26/16.
  */
class DenseLayer(_output_dim: Int,
                 _input_dim: Int = -1,
                 activation: String = "linear",
                 var _weights: DenseMatrix[Double] = null,
                 var _bias: DenseVector[Double] = null,
                 var _use_bias: Boolean = true,
                 _name: String = null) extends ActivationLayer(_input_dim, _output_dim, activation, _name)
{

  override def weights : DenseMatrix[Double] = {
    return if (_use_bias) DenseMatrix.vertcat(weights, _bias.asDenseMatrix) else weights
  }

  override def weights_= (that: DenseMatrix[Double]) { _weights = that }

  /** Predict with a single sample of training data. */
  override def predict(inputs : DenseVector[Double]) : Double = {
    val cell_body : DenseVector[Double] = (inputs.t * _weights).t
    if (_use_bias) cell_body += _bias
    return super.step(sum(cell_body))
  }

  /** Predict with multiple samples of training data. */
  override def predict(inputs : DenseMatrix[Double]) : DenseVector[Double] = {
    return inputs(*, ::).map { row => predict(row) }
  }
}

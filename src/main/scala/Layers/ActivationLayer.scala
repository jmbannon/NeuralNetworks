package Layers

import Functions.{Activation, Activations}
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}

import scala.annotation.meta.{param, setter}

/**
  * Created by jb on 9/30/16.
  */
class ActivationLayer(_output_dim: Int,
                      _input_dim: Int = -1,
                      val activation: String = "linear",
                      _name: String = null) extends Layer(_output_dim, _input_dim, _name)
{
  require(Activations.exists(activation.toLowerCase), "Activation function " + activation + " does not exist")

  private val _activation : Activation = Activations(activation)

  @param def step : Activation = { _activation }
  @param def weights = DenseMatrix.fill(_output_dim, _input_dim)(0.0)

  @setter def weights_= (tmp: DenseMatrix[Double]) { }


  /** Predict with a single sample of training data. */
  def predict(inputs : DenseVector[Double]) : Double = {
    return _activation(sum(inputs))
  }

  /** Predict with multiple samples of training data. */
  def predict(inputs : DenseMatrix[Double]) : DenseVector[Double] = {
    return inputs(*, ::).map { row => predict(row) }
  }
}

package Models

import scala.collection.mutable.ListBuffer
import Functions.{Cost, Costs}
import Layers.ActivationLayer
import breeze.linalg.{*, DenseMatrix, DenseVector}

/**
  * Created by jb on 9/30/16.
  */
class Sequential(val _layers: ListBuffer[ActivationLayer] = ListBuffer()) {

  var _cost: Cost = null
  var _nr_layers: Int = _layers.size

  def add(layer: ActivationLayer) = {
    _layers.append(layer)
  }

  def compile(cost: String) = {
    require(Costs.exists(cost.toLowerCase), "Cost function " + cost + " does not exist")
    require(_layers(0).input_shape > 0, "Must specify input shape on input layer")

    _cost = Costs(cost)
    _nr_layers = _layers.size

    for (i <- 1 to (_nr_layers - 1)) {
        _layers(i)._input_shape = _layers(i - 1)._output_shape
    }

    for (i <- 0 to (_nr_layers - 1)) {
      if (_layers(i).weights != null) {
        //require(_layers(i).weights.rows )
      }
    }
  }

  private def infer(input_data: DenseMatrix[Double],
                    verbose: Boolean = false): Array[DenseVector[Double]] = {

    val layer_outputs = Array.ofDim[DenseVector[Double]](_layers.size)
    layer_outputs(0) := _layers(0).predict(input_data)

    if (verbose) {
      println("Input Data:")
      println(input_data.toString())
      println("------------------------------")
      println(_layers(0).display_name(0))
      println(layer_outputs(0).toString)
    }

    for (i <- 1 to _layers.size - 1) {
      layer_outputs(i) := _layers.apply(i).predict(layer_outputs(i - 1))
      if (verbose) {
        println("------------------------------")
        println(_layers(i).display_name(i))
        println(layer_outputs(i).toString())
      }
    }
    return layer_outputs
  }

  def fit(train: DenseMatrix[Double],
          targets: DenseVector[Double],
          labels: Array[String] = null,
          nr_epoch: Int = 100,
          batch_size: Int = 0,
          learning_rate: Double = 0.001,
          verbose: Boolean = false) = {
    val deltas = Array.ofDim[DenseVector[Double]](_nr_layers)
    val errors = Array.ofDim[DenseVector[Double]](_nr_layers)
    val shift = Array.ofDim[DenseVector[Double]](_nr_layers)

    var outputs = Array.ofDim[DenseVector[Double]](_nr_layers)
    val output_layer : Int = _nr_layers - 1

    for (iter <- 1 to nr_epoch) {
      outputs = infer(train, verbose)
      errors(output_layer) := this._cost.d(outputs(output_layer), targets)
      deltas(output_layer) := errors(output_layer) :* this._layers(output_layer).step.d(outputs(output_layer))
      shift(output_layer) := deltas(output_layer).dot(outputs(output_layer - 1))

      for (i <- (output_layer - 1) to 0) {
        errors(i) := (deltas(i + 1).t * (_layers(i + 1).weights)).t
        deltas(i) := errors(i) :* this._layers(i).step.d(outputs(i))

        if (i != 0) {
          shift(i) := deltas(i).dot(outputs(i - 1))
        } else {
          shift(i) := (deltas(i).t * train).t
        }
      }

      for (i <- 0 to output_layer) {
        _layers(i).weights_= (_layers(i).weights(*,::) + (-learning_rate * shift(i)))
      }
    }
  }

}

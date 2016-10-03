package Models

import scala.collection.mutable.ListBuffer
import Functions.{Cost, Costs}
import Layers.DenseLayer
import breeze.linalg.{*, DenseMatrix, DenseVector}

/**
  * Created by jb on 9/30/16.
  */
class Sequential(val _layers: ListBuffer[DenseLayer] = ListBuffer()) {

  var _cost: Cost = null
  var _nr_layers: Int = _layers.size

  def add(layer: DenseLayer) = {
    _layers.append(layer)
  }

  def compile(cost: String) = {
    require(Costs.exists(cost.toLowerCase), "Cost function " + cost + " does not exist")
    require(_layers(0).input_shape > 0, "Must specify input shape on input layer")

    _cost = Costs(cost)
    _nr_layers = _layers.size

    for (i <- 1 to (_nr_layers - 1)) {
      _layers(i)._input_shape = _layers(i - 1).output_shape
    }

    for (i <- 0 to (_nr_layers - 1)) {
      println("Generating weights for layer " + i)
      if (!_layers(i).has_weights) {
        _layers(i).gen_weights
      } else {
        require(_layers(i).weights.rows == _layers(i).input_shape, "Weight nr_rows must match input shape")
        require(_layers(i).weights.cols == _layers(i).output_shape, "Weight nr_cols must match output shape")
      }
    }
  }

  private def infer(input_data: DenseMatrix[Double],
                    verbose: Boolean = false): Array[DenseMatrix[Double]] = {

    val layer_outputs = Array.ofDim[DenseMatrix[Double]](_layers.size)

    layer_outputs(0) = _layers(0).predict(input_data)

    if (verbose) {
      println("Input Data:")
      println(input_data.toString())
      println("------------------------------")
      println(_layers(0).display_name(0))
      println(layer_outputs(0).toString)
    }

    for (i <- 1 to _layers.size - 1) {
      layer_outputs(i) = _layers(i).predict(layer_outputs(i - 1))
      if (verbose) {
        println("------------------------------")
        println(_layers(i).display_name(i))
        println(layer_outputs(i).toString())
      }
    }
    return layer_outputs
  }

  def fit(train: DenseMatrix[Double],
          targets: DenseMatrix[Double],
          labels: Array[String] = null,
          nr_epoch: Int = 100,
          batch_size: Int = 0,
          learning_rate: Double = 0.001,
          verbose: Boolean = false) = {

    val deltas = Array.ofDim[DenseMatrix[Double]](_nr_layers)
    val errors = Array.ofDim[DenseMatrix[Double]](_nr_layers)
    val shift = Array.ofDim[DenseMatrix[Double]](_nr_layers)

    var outputs = Array.ofDim[DenseMatrix[Double]](_nr_layers)
    val output_layer : Int = _nr_layers - 1

    for (iter <- 1 to nr_epoch) {
      var i : Int = output_layer
      outputs = infer(train, verbose)
//      println("we inferred!!!")
      println("Outputs:")
      println(outputs(i))
      println("Expected outputs:")
      println(targets)

      errors(i) = this._cost.d(outputs(i), targets).toDenseMatrix
//      println("errors")
//      println(errors(i))
//      println(this._layers(i).step.d(outputs(i)))
      deltas(i) = errors(i) :* this._layers(i).gradient(outputs(i))
//      println("deltas")
//      println(deltas(i))
      shift(i) = outputs(i - 1) * deltas(i)
//      println("shift")
//      println(shift(i))

      for (i <- (output_layer - 1) to 0) {
        errors(i) = deltas(i + 1) * _layers(i + 1).weights.t
        deltas(i) = errors(i) :* this._layers(i).gradient(outputs(i))

        if (i != 0) {
          shift(i) = deltas(i) * (outputs(i - 1))
        } else {
          shift(i) = (deltas(i).t * train).t
        }
      }

      for (i <- 0 to output_layer) {
        this._layers(i).weights_= (this._layers(i).weights + (-learning_rate * shift(i)))
      }
    }
  }

}

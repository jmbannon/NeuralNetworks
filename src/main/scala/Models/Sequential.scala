package Models

import scala.collection.mutable.ListBuffer
import Functions.{Cost, Costs}
import Layers.DenseLayer
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}

/**
  * Created by jb on 9/30/16.
  */
class Sequential(val _layers: ListBuffer[DenseLayer] = ListBuffer()) {
  var _cost: Cost = null

  def add(layer: DenseLayer) = {
    _layers.append(layer)
  }

  def compile(cost: String) = {
    require(Costs.exists(cost.toLowerCase), "Cost function " + cost + " does not exist")
    _cost = Costs(cost)

    for (i <- _layers.indices) {
      if (i != 0) {
        _layers(i)._input_shape = _layers(i - 1).output_shape
      } else {
        require(_layers(0).input_shape._is1D, "Must specify 1-D input shape on input layer")
      }

      if (!_layers(i).contains_weights) {
        _layers(i).gen_weights
      } else {
        require(_layers(i).weights.rows == _layers(i).input_shape, "Weight nr_rows must match input shape")
        require(_layers(i).weights.cols == _layers(i).output_shape, "Weight nr_cols must match output shape")
      }
    }
  }

  def predict(input_data: DenseMatrix[Double],
              verbose: Boolean = false): DenseMatrix[Double] = {
    val output_layer: Int = _layers.size - 1
    this.feed_forward(input_data, verbose)(output_layer)
  }

  private def feed_forward(input_data: DenseMatrix[Double],
                           verbose: Boolean = false): Array[DenseMatrix[Double]] = {

    val layer_outputs = Array.ofDim[DenseMatrix[Double]](_layers.size)

    if (verbose) {
      println("------------------------------")
      println("Feed forward")
    }
    for (i <- _layers.indices) {
      if (i != 0) {
        layer_outputs(i) = _layers(i).step(layer_outputs(i - 1))
      } else {
        layer_outputs(i) = _layers(i).step(input_data)
      }
      if (verbose) {
        println(_layers(i).display_name(i))
        println("Weights")
        println(_layers(i).weights)
        println("Output")
        println(layer_outputs(i).toString())
        println("------------------------------")
      }
    }
    return layer_outputs
  }

  def fit(train: DenseMatrix[Double],
          targets: DenseMatrix[Double],
          labels: Array[String] = null,
          nr_epoch: Int = 100,
          batch_size: Int = 0,
          learning_rate: Double = 0.1,
          verbose: Boolean = false) = {

    val deltas = Array.ofDim[DenseMatrix[Double]](_layers.size)
    val errors = Array.ofDim[DenseMatrix[Double]](_layers.size)
    val shift = Array.ofDim[DenseMatrix[Double]](_layers.size)

    var outputs = Array.ofDim[DenseMatrix[Double]](_layers.size)
    val output_layer : Int = _layers.size - 1

    for (iter <- 1 to nr_epoch) {
      outputs = feed_forward(train, verbose)
      if (verbose) {
        println("Outputs | Expected Outputs")
        for (samples <- 0 to (targets.rows - 1)) {
          print(outputs(output_layer)(samples, ::).inner)
          print("   ")
          print(targets(samples, ::).inner)
          print("\n")
        }
        println("------------------------------")
      }

      for (i <- output_layer to 0 by -1) {
        if (i == output_layer) {
          errors(i) = this._cost.d(outputs(i).copy, targets.copy).toDenseMatrix
        } else {
          errors(i) = deltas(i + 1).copy * _layers(i + 1).weights.copy.t
        }
        if (verbose) {
          println("Errors")
          println(errors(i))
          println("------------------------------")
        }

        deltas(i) = errors(i).copy :* this._layers(i).gradient(outputs(i).copy)
        if (verbose) {
          println("Deltas")
          println(deltas(i))
          println("------------------------------")
        }

        if (i != 0) {
          shift(i) = outputs(i - 1).t * deltas(i).copy
        } else {
          shift(i) = train.copy.t * deltas(i).copy
        }
        if (verbose) {
          println("Shift")
          println(shift(i))
          println("------------------------------")
        }
      }

      for (i <- 0 to output_layer) {
        this._layers(i).updateWeights(learning_rate * shift(i).copy)
        if (this._layers(i).use_bias) {
          val deltaBiasVec = sum(deltas(i)(::,*)).t
          this._layers(i).updateBias(learning_rate * deltaBiasVec)
        }
      }
    }
    outputs = feed_forward(train, verbose)
    println("Outputs | Expected Outputs")
    for (samples <- 0 to (targets.rows - 1)) {
      print(outputs(output_layer)(samples, ::).inner)
      print("   ")
      print(targets(samples, ::).inner)
      print("\n")
    }
    println("------------------------------")
  }
}

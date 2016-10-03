import Layers.DenseLayer
import breeze.linalg.{DenseMatrix, DenseVector}
import Models.Sequential

/**
  * Created by jb on 10/1/16.
  */
object Test {
  def main(args: Array[String]) {
    val data : DenseMatrix[Double] = DenseMatrix((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    val output = DenseMatrix(0.0, 1.0, 1.0, 0.0)

    val model = new Sequential()
    model.add(new DenseLayer(4, _input_dim = 2, activation = "sigmoid", _use_bias = true))
    model.add(new DenseLayer(1, activation = "sigmoid", _use_bias = true))
    model.compile(cost = "quadratic")
    model.fit(data, output, nr_epoch = 100, verbose = false)

  }
}
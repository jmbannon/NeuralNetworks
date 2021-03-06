import Layers.{DenseLayer, Shape}
import NNMath.Volume
import breeze.linalg.{DenseMatrix, DenseVector}
import Models.Sequential

/**
  * Created by jb on 10/1/16.
  */
object Test {
  def main(args: Array[String]) {
    val data : DenseMatrix[Double] = DenseMatrix(
      (0.0, 0.0, 0.0),
      (0.0, 1.0, 1.0),
      (1.0, 0.0, 0.0),
      (1.0, 1.0, 1.0),
      (1.0, 0.0, 1.0))

    val output = DenseMatrix(0.0, 1.0, 1.0, 0.0, 1.0)

    val model = new Sequential()
    model.add(DenseLayer(4, input_dim = 3, activation = "sigmoid", use_bias = true))
    model.add(DenseLayer(8, activation = "sigmoid", use_bias = true))
    model.add(DenseLayer(1, activation = "sigmoid", use_bias = true))
    model.compile(cost = "quadratic")
    model.fit(data, output, nr_epoch = 100000, verbose = false, learning_rate = 0.5)

    println("meh")
    println(model.predict(DenseVector(0.0, 1.0, 1.0).toDenseMatrix))
    println(model.predict(DenseVector(1.0, 0.0, 1.0).toDenseMatrix))
    println(model.predict(DenseVector(0.0, 1.0, 0.0).toDenseMatrix))

    println("VOLUME TEST")
    volumeTest()
  }

  def volumeTest() {
    val layers: Int = 5
    val nrows: Int = 10
    val ncols: Int = 15
    val vol = Volume.rand(nrows, ncols, layers)

    println(vol.toString())
    println(vol(0 to 2, 0 to 2).toString)
  }
}
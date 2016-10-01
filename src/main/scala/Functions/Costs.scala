package Functions

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.numerics.{log, pow}

import scala.collection.immutable.HashMap

/**
  * Created by jb on 9/30/16.
  */
object Costs {
  private val _quadratic = new Cost("quadratic") {
    override def apply(output: Double, exp_output: Double): Double = {
      0.5 * Math.pow(output - exp_output, 2)
    }
    override def d(output: Double, exp_output: Double): Double = {
      output - exp_output
    }
    override def apply(output: DenseVector[Double], exp_output: DenseVector[Double]): Double = {
      0.5 * sum(pow(output - exp_output, 2))
    }
    override def d(output: DenseVector[Double], exp_output: DenseVector[Double]): DenseVector[Double] = {
      output - exp_output
    }
  }

  private val _crossentropy = new Cost("crossentropy") {
    override def apply(output: Double, exp_output: Double): Double = {
      (exp_output * log(output)) + ((1.0 - exp_output) * log(1.0 - output))
    }
    override def d(output: Double, exp_output: Double): Double = {
      (output - exp_output) / ((output + 1.0) * output)
    }
    override def apply(output: DenseVector[Double], exp_output: DenseVector[Double]): Double = {
      -sum((exp_output :* log(output)) + ((1.0 - exp_output) :* log(1.0 - output)))
    }
    override def d(output: DenseVector[Double], exp_output: DenseVector[Double]): DenseVector[Double] = {
      (output - exp_output) / ((output + 1.0) :* output)
    }
  }

  private val _relativeentropy = new Cost("relativeentropy") {
    override def apply(output: Double, exp_output: Double): Double = {
      exp_output * Math.log(exp_output / output)
    }
    override def d(output: Double, exp_output: Double): Double = {
      exp_output / output
    }
    override def apply(output: DenseVector[Double], exp_output: DenseVector[Double]): Double = {
      sum(exp_output :* log(exp_output :/ output))
    }
    override def d(output: DenseVector[Double], exp_output: DenseVector[Double]): DenseVector[Double] = {
      exp_output / output
    }
  }

  private val costs : HashMap[String, Cost] = HashMap(
    _quadratic.name -> _quadratic,
    _crossentropy.name -> _crossentropy,
    _relativeentropy.name -> _relativeentropy
  )

  def apply(name : String) : Cost = {
    costs.get(name).get
  }

  def exists(name : String) : Boolean = {
    costs.isDefinedAt(name)
  }
}

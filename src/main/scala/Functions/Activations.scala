package Functions

import breeze.linalg.DenseVector
import breeze.numerics.{exp, pow, tanh}

import scala.collection.immutable.HashMap

/**
  * Created by jb on 9/30/16.
  */
object Activations {

  private val _sigmoid = new Activation("sigmoid") {
    override def apply(input: Double): Double = {
      1.0/ (1.0 + Math.exp(-input))
    }
    override def d(input: Double): Double = {
      input * (1 - input)
    }
  }

  private val _tanh = new Activation("tanh") {
    override def apply(input: Double): Double = {
      Math.tanh(input)
    }
    override def d(input: Double): Double = {
      1.0 - (input * input)
    }
  }

  private val _linear = new Activation("linear") {
    override def apply(input: Double): Double = {
      input
    }
    override def d(input: Double): Double = {
      1.0
    }
  }

  private val _binarystep = new Activation("binarystep") {
    override def apply(input: Double): Double = {
      if (input >= 0) 1.0 else 0.0
    }
    override def d(input: Double): Double = {
      0.0
    }
  }

  private val activations : HashMap[String, Activation] = HashMap(
    _sigmoid.name -> _sigmoid,
    _tanh.name -> _tanh,
    _linear.name -> _linear,
    _binarystep.name -> _binarystep)

  def apply(name : String) : Activation = {
    activations.get(name).get
  }

  def exists(name : String) : Boolean = {
    activations.isDefinedAt(name)
  }
}

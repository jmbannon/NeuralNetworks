package Functions

import scala.collection.immutable.HashMap

/**
  * Created by jb on 9/30/16.
  */
object Activations {

  private val _sigmoid = new Activation("sigmoid") {
    override def apply(x: Double): Double = { 1.0/ (1.0 + Math.exp(-x)) }
    override def d(x: Double): Double = { x * (1 - x) }
  }

  private val _tanh = new Activation("tanh") {
    override def apply(x: Double): Double = { Math.tanh(x) }
    override def d(x: Double): Double = { 1.0 - (x * x) }
  }

  private val _linear = new Activation("linear") {
    override def apply(x: Double): Double = { x }
    override def d(x: Double): Double = { 1.0 }
  }

  private val _binary_step = new Activation("binary_step") {
    override def apply(x: Double): Double = { if (x >= 0) 1.0 else 0.0 }
    override def d(x: Double): Double = { 0.0 }
  }

  private val _relu = new Activation("relu") {
    override def apply(x: Double): Double = {  if (x >= 0) x else 0.0 }
    override def d(x: Double): Double = { if (x >= 0) 1.0 else 0.0 }
  }

  private val _arctan = new Activation("arctan") {
    override def apply(x: Double): Double = { Math.atan(x) }
    override def d(x: Double): Double = { 1.0 / ((x * x) + 1)}
  }

  private val _bent_identity = new Activation("bent_identity") {
    override def apply(x: Double): Double = { (Math.sqrt((x * x) + 1.0) - 1.0) / 2.0 }
    override def d(x: Double): Double = { (x / (2 * Math.sqrt((x * x) + 1))) + 1 }
  }

  private val activations : HashMap[String, Activation] = HashMap(
    _sigmoid.name -> _sigmoid,
    _tanh.name -> _tanh,
    _linear.name -> _linear,
    _binary_step.name -> _binary_step,
    _relu.name -> _relu,
    _arctan.name -> _arctan,
    _bent_identity.name -> _bent_identity)

  def apply(name : String) : Activation = {
    activations.get(name).get
  }

  def exists(name : String) : Boolean = {
    activations.isDefinedAt(name)
  }
}

package scarletbrain

import scarletbrain.FeedForwardNetwork.{Gradient, Layer}
import scarletbrain.LinearAlgebra.{Matrix, Vector}

import scala.util.Random

case class FeedForwardNetwork(layers: List[Layer]) {
  def evaluate(in: Vector): Vector = {
    layers.foldLeft(in) { case (input, layer) =>
      (layer.weight rowProdSums input) + layer.bias
    }
  }

  // Backpropagate, returning calculated gradient for this example and the cost for this example
  def backprop(input: Vector, output: Vector): (Gradient, Double) = {
    def rec(list: List[Layer], in: Vector, out: Vector): (List[(Matrix, Vector)], Vector, Double) = list match {
      case Nil =>
        val diff = in - out
        (Nil, diff, (diff * diff).sum)
      case Layer(weight, bias, activation) :: tail =>
        // Evaluate this layer's nodes using last layer's in
        val a = (weight rowProdSums in) + bias

        // Recursively backprop further layers, passing this evaluated layer to the next
        val (next, dCda, cost) = rec(tail, a, out)

        // Calculate this layer's gradient
        val dadz = a.map(activation.deriv)
        val dCdz = dCda * dadz

        val dCdb = dCdz
        val dCdw = dCdz prodMatrix in
        val dCda_1 = weight colProdSums dCda

        ((dCdw, dCdb) :: next, dCda_1, cost)
    }

    val (g, _, c) = rec(layers, input, output)
    (Gradient(g), c)
  }

  def train(batch: Iterable[(Vector, Vector)], step: Double): (FeedForwardNetwork, Double) = {
    val gs = batch.par.map{
      case (in, out) => backprop(in, out)
    }

    val avgCost = gs.map(_._2).sum / batch.size
    val avgGrad = gs.map(_._1).reduce(_ + _) * (step/batch.size)

    val newLayers = layers.zip(avgGrad.grads).map {
      case (Layer(w, b, act), (wg, bg)) => Layer(w - wg, b - bg, act)
    }

    if (avgCost == Double.NaN) {
      println("Ugh")
    }

    (FeedForwardNetwork(newLayers), avgCost)
  }

  override def toString: String = layers.mkString("\n\n")
}

object FeedForwardNetwork {
  def fillRandom(layerSizes: Seq[Int], act: Activation): FeedForwardNetwork =
    new FeedForwardNetwork(
      /* Sliding view of 2 elements at a time to form each layer */
      layerSizes.sliding(2).map { case Seq(prevSize, thisSize) =>
          Layer.fillRandom(prevSize, thisSize, act)
      }.toList
    )

  case class Gradient(grads: List[(Matrix, Vector)]) extends AnyVal {
    def +(g: Gradient): Gradient = Gradient(grads.zip(g.grads).map{
      case ((w1, b1), (w2, b2)) => (w1+w2, b1+b2)
    })

    def *(s: Double): Gradient = Gradient(grads.map{case (w, b) => (w*s, b*s) })
  }

  case class Layer(weight: Matrix, bias: Vector, act: Activation) {
    override def toString: String = s"Weights: $weight\nBiases: $bias\nActivation: $act"
  }
  object Layer {
    // Makes Layer with random weights and biases (Gaussian standard distribution)
    def fillRandom(prevSize: Int, thisSize: Int, act: Activation): Layer = {
      // Makes random arrays
      def randArray(size: Int): Array[Double] = {
        val a = new Array[Double](size)
        for (i <- a.indices) a(i) = Random.nextGaussian() // Fill with random gaussian values
        a
      }
      Layer(
        weight = Matrix(randArray(prevSize * thisSize), thisSize, prevSize),
        bias = Vector(randArray(thisSize)),
        act
      )
    }
  }
}

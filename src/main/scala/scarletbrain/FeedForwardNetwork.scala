package scarletbrain

import scarletbrain.FeedForwardNetwork.Layer
import scarletbrain.LinearAlgebra._

import scala.util.Random

case class FeedForwardNetwork(layers: List[Layer]) {
  def evaluate(in: Vector): Vector = {
    layers.foldLeft(in) { case (input, Layer(weight, bias, act)) =>
      ((weight rowProdSums input) + bias).map(act(_))
    }
  }

  override def toString: String = s"FeedForwardNetwork(\n${layers.mkString(" ::\n")} :: Nil)"
}

object FeedForwardNetwork {
  def fillRandom(layerSizes: Seq[Int], act: Activation): FeedForwardNetwork =
    new FeedForwardNetwork(
      /* Sliding view of 2 elements at a time to form each layer */
      layerSizes.sliding(2).map { case Seq(prevSize, thisSize) =>
          Layer.fillRandom(prevSize, thisSize, act)
      }.toList
    )

  case class Layer(weight: Matrix, bias: Vector, act: Activation) {
    override def toString: String = s"Layer(weight = $weight,\n      bias = $bias,\n      act = $act)"
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

package scarletbrain

import scarletbrain.BackPropNetwork.Gradient
import scarletbrain.FeedForwardNetwork.Layer
import scarletbrain.LinearAlgebra._

import scala.annotation.tailrec

class BackPropNetwork(override val layers: List[Layer]) extends FeedForwardNetwork(layers) {

  // Cost squared
  def cost(in: Vector, out: Vector): Double = (evaluate(in) - out).mag2

  def batchCost(batch: Iterable[(VectorGeneric, VectorGeneric)]): Double =
    batch.par.map{case (in, out) => cost(in, out)}.sum / batch.size

  // Backpropagate, returning calculated gradient for this example and the cost2 for this example
  def backprop(input: Vector, output: Vector): (Gradient, Double) = {
    def rec(list: List[Layer], in: Vector, out: Vector): (List[(Matrix, VectorGeneric)], Vector, Double) = list match {
      case Nil =>
        val diff = in - out
        (Nil, diff, diff.mag2)
      case Layer(weight, bias, act) :: tail =>
        // Evaluate this layer's nodes using last layer's in
        val a = ((weight rowProdSums in) + bias).map(act(_))

        // Recursively backprop further layers, passing this evaluated layer to the next
        val (next, dCda, cost) = rec(tail, a, out)

        // Calculate this layer's gradient
        val dadz = a.map(act.deriv)
        val dCdz = dCda * dadz

        val dCdb = dCdz
        val dCdw = dCdz prodMatrix in
        val dCda_1 = weight colProdSums dCda

        ((dCdw, dCdb: VectorGeneric) :: next, dCda_1, cost)
    }

    val (g, _, c) = rec(layers, input, output)
    (Gradient(g), c)
  }

  def batchBackprop(batch: Iterable[(VectorGeneric, VectorGeneric)]): (Gradient, Double) = {
    val gs = batch.par.map{
      case (in, out) => backprop(in, out)
    }

    val avgCost = gs.map(_._2).sum / batch.size
    val avgGrad = gs.map(_._1).reduce(_ + _) * (1.0/batch.size)

    (avgGrad, avgCost)
  }

  def -(grad: Gradient): BackPropNetwork =
    BackPropNetwork(layers.zip(grad.grads).map {
      case (Layer(w, b, act), (wg, bg)) => Layer(w - wg, b - bg, act)
    })
}

object BackPropNetwork {
  def apply(layers: List[Layer]): BackPropNetwork = new BackPropNetwork(layers)

  case class Gradient(grads: List[(Matrix, VectorGeneric)]) extends AnyVal {
    def +(g: Gradient): Gradient =
      Gradient(grads.zip(g.grads).map{
        case ((w1, b1), (w2, b2)) => (w1+w2, b1+b2: VectorGeneric)
      })

    def *(s: Double): Gradient = Gradient(grads.map{case (w, b) => (w*s, b*s: VectorGeneric) })

    def mag2: Double = {
      @tailrec
      def rec(list: List[(Matrix, VectorGeneric)], acc: Double): Double = list match {
        case Nil => acc
        case (m, v) :: tail => rec(tail, acc + m.mag2 + v.mag2)
      }
      rec(grads, 0)
    }
  }


  trait Trainer {
    def step(net: BackPropNetwork, batch: Iterable[(VectorGeneric, VectorGeneric)]): (BackPropNetwork, Double)
  }

  object Trainer {

    case class ConstantStep(step: Double) extends Trainer {
      override def step(net: BackPropNetwork, batch: Iterable[(VectorGeneric, VectorGeneric)]): (BackPropNetwork, Double) = {
        val (avgGrad, avgCost) = net.batchBackprop(batch)

        (net - (avgGrad * step), avgCost)
      }
    }

    case class BacktrackLine(maxStep: Double, minStep: Double, stepMult: Double, slopeFactor: Double) extends Trainer {
      override def step(net: BackPropNetwork, batch: Iterable[(VectorGeneric, VectorGeneric)]): (BackPropNetwork, Double) = {

        val (avgGrad, avgCost) = net.batchBackprop(batch)

        // Calculate step size using backtracking
        val expectedSlope = Math.sqrt(avgGrad.mag2) * slopeFactor

        var step = maxStep
        var newNet = net - (avgGrad * step)

        while (step > minStep && newNet.batchCost(batch) > avgCost - expectedSlope * step) {
          step *= stepMult
          newNet = net - (avgGrad * step)
        }

        (newNet, avgCost)
      }
    }

  }
}
package scarletbrain

import scarletbrain.Activation.{Identity, Logistic}
import scarletbrain.BackPropNetwork.Trainer
import scarletbrain.FeedForwardNetwork.Layer
import scarletbrain.LinearAlgebra.{Matrix, Vector, VectorGeneric}

import scala.util.Random

object NetTrainTest {
  def main(args: Array[String]): Unit = {
    val training = (1 to 1000).map { _ =>
      val v = Vector(Random.nextInt(2), Random.nextInt(2), Random.nextInt(2), Random.nextInt(2)): VectorGeneric
      (v, v)
    }

    val batch = Iterator.continually(Random.shuffle(training).grouped(50)).flatten

    var net = BackPropNetwork(
      Layer(weight = Matrix(rows = 2, cols = 4, -2.6328747055461923,-4.971960543686017,2.733932788962138,-5.16668364198991,4.67411327076069,-2.914473103369734,-4.8511331109772255,-3.0282193719910397),
        bias = Vector(5.385902355859904, 3.4560021975510624),
        act = Identity) ::
        Layer(weight = Matrix(rows = 2, cols = 2, -1.0797601275703197,2.494831377747256,2.3095601999185775,1.1165809675814802),
          bias = Vector(-0.9726603352810441, -1.4598284190609137),
          act = Identity) ::
        Layer(weight = Matrix(rows = 4, cols = 2, 0.31147792270144287,-0.05091457884965522,-0.016381260072725973,-0.3049988885487813,-15.819873603105743,1.9033069663676667,-0.9054740974408652,-16.49018899504059),
          bias = Vector(-0.020642425998781584, 0.10981994777413957, -6.037063181133185, -3.327471299093861),
          act = Logistic) :: Nil)

    println(net.evaluate(Vector(0,1,1,0)))
    println(net.evaluate(Vector(1,0,0,0)))



    val trainer = Trainer.ConstantStep(0.05) //BacktracingLine(maxStep = 10.0, minStep = 1E-7, stepMult = 0.8, slopeFactor = 0.5)

    var lastC = 1.0
    while(lastC > 0.2) {

      lastC = 0.0

      for (_ <- 0 until 1000) {
        val trained = trainer.step(net, batch.next())
        net = trained._1
        lastC += trained._2
      }

      lastC /= 1000.0

      println(lastC)
      println(net)

    }

    println(net)

    println(net.evaluate(Vector(0,1,1,0)))
    println(net.evaluate(Vector(1,0,0,0)))
  }

}

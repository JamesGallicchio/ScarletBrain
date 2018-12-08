package scarletbrain

import scarletbrain.LinearAlgebra._

import scala.reflect.io.{File, Path}
import scala.util.Random

object ScarletBrain {
  def main(args: Array[String]): Unit = {

    val training = (1 to 1000).map { _ =>
      val (a,b) = (Random.nextInt(100), Random.nextInt(100))
      (Vector(a,b),Vector(2*a + b))
    }

    var net = FeedForwardNetwork.fillRandom(Seq(2 ,1), Activation.Identity)
    var step = 1E-5

    var lastC = 100.0

    val file = Path("rnndata.csv").toFile.bufferedWriter()

    while(lastC > 0.001) {
      val trained = net.train(Random.shuffle(training).take(100), step)
      net = trained._1
      lastC = trained._2

      file.write(lastC.toString)
      file.newLine()
    }

    println(net)
    println(lastC)
  }
}

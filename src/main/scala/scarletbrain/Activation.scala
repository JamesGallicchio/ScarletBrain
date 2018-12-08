package scarletbrain

trait Activation {
  def apply(x: Double): Double
  def deriv(x: Double): Double
}
object Activation {

  case object Identity extends Activation {
    override def apply(x: Double): Double = x
    override def deriv(x: Double): Double = 1.0
  }

  case object Step extends Activation {
    override def apply(x: Double): Double = if (x < 0.0) 0.0 else 1.0
    override def deriv(x: Double): Double = if (x != 0.0) 0.0 else Double.PositiveInfinity
  }

  case object Logistic extends Activation {
    override def apply(x: Double): Double = 1.0/(1.0 + Math.exp(-x))
    override def deriv(x: Double): Double = {
      val s = Logistic(x)
      s * (1.0 - s)
    }
  }

  case object TanH extends Activation {
    override def apply(x: Double): Double = Math.tanh(x)
    override def deriv(x: Double): Double = {
      val t = Math.tanh(x)
      1 - t*t
    }
  }

  case object ArcTan extends Activation {
    override def apply(x: Double): Double = Math.atan(x)
    override def deriv(x: Double): Double = 1.0/(1.0 + x*x)
  }

  case object ReLU extends Activation {
    override def apply(x: Double): Double = if (x < 0.0) 0.0 else x
    override def deriv(x: Double): Double = if (x < 0.0) 0.0 else 1.0
  }

  case class PReLU(alpha: Double) extends Activation {
    override def apply(x: Double): Double = if (x < 0.0) alpha * x else 1.0
    override def deriv(x: Double): Double = if (x < 0.0) alpha else 1.0
  }
}

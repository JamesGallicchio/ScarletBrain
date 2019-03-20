package scarletbrain

object LinearAlgebra {

  type VectorTag
  type VectorGeneric = Array[Double] with VectorTag
  implicit val VecGen2Vec: VectorGeneric => Vector = Vector(_)
  implicit val Vec2VecGen: Vector => VectorGeneric = _.arr.asInstanceOf[VectorGeneric]

  case class Vector(arr: Array[Double]) {
    def +(v: Vector): Vector = if (arr.length != v.arr.length) lengthError() else {
      val other = v.arr
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) + other(i) ; i += 1 }
      Vector(a)
    }

    def +(s: Double): Vector = {
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) + s ; i += 1 }
      Vector(a)
    }

    def -(v: Vector): Vector = if (arr.length != v.arr.length) lengthError() else {
      val other = v.arr
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) - other(i) ; i += 1 }
      Vector(a)
    }

    def -(s: Double): Vector = {
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) - s ; i += 1 }
      Vector(a)
    }

    def *(v: Vector): Vector = if (arr.length != v.arr.length) lengthError() else {
      val other = v.arr
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) * other(i) ; i += 1 }
      Vector(a)
    }

    def *(s: Double): Vector = {
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) * s ; i += 1 }
      Vector(a)
    }

    def prodMatrix(v: Vector): Matrix = {
      val a = new Array[Double](arr.length * v.arr.length)
      var i = 0
      while (i < arr.length) {
        val offset = i * v.arr.length
        val mult = arr(i)
        var j = 0
        while (j < v.arr.length) {
          a(offset + j) = mult * v.arr(j)
          j += 1
        }
        i += 1
      }
      Matrix(a, arr.length, v.arr.length)
    }

    def map(f: Double => Double): Vector = {
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = f(arr(i)) ; i += 1 }
      Vector(a)
    }

    lazy val sum: Double = {
      var total = 0.0
      var i = 0
      while (i < arr.length) {
        total += arr(i)
        i += 1
      }
      total
    }

    lazy val mag2: Double = {
      var total = 0.0
      var i = 0
      while (i < arr.length) {
        total += arr(i) * arr(i)
        i += 1
      }
      total
    }

    lazy override val toString: String =
      s"Vector(${arr.mkString(", ")})"
  }
  object Vector {
    def apply(values: Double*): Vector = new Vector(values.toArray)
  }

  case class Matrix private (private[LinearAlgebra] val arr: Array[Double], rows: Int, cols: Int) {
    def +(mx: Matrix): Matrix = if (arr.length != mx.arr.length) lengthError() else {
      val other = mx.arr
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) + other(i) ; i += 1 }
      Matrix(a, rows, cols)
    }

    def +(s: Double): Matrix = {
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) + s ; i += 1 }
      Matrix(a, rows, cols)
  }

    def -(mx: Matrix): Matrix = if (arr.length != mx.arr.length) lengthError() else {
      val other = mx.arr
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) - other(i) ; i += 1 }
      Matrix(a, rows, cols)
    }

    def rowProdSums(vec: Vector): Vector = if (vec.arr.length != cols) lengthError() else {
      val other = vec.arr
      val a = new Array[Double](rows)
      var i = 0
      while (i < rows) {
        val offset = i * cols
        var j = 0
        while (j < cols) {
          a(i) += arr(offset + j) * other(j)
          j += 1
        }
        i += 1
      }
      Vector(a)
    }

    def colProdSums(vec: Vector): Vector = if (vec.arr.length != rows) lengthError() else {
      val other = vec.arr
      val a = new Array[Double](cols)
      var i = 0
      while (i < rows) {
        val offset = i * cols
        var j = 0
        while (j < cols) {
          a(j) += arr(offset + j) * other(i)
          j += 1
        }
        i += 1
      }
      Vector(a)
    }

    def *(s: Double): Matrix = {
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) * s; i += 1 }
      Matrix(a, rows, cols)
    }

    lazy val mag2: Double = {
      var total = 0.0
      var i = 0
      while (i < arr.length) {
        total += arr(i) * arr(i)
        i += 1
      }
      total
    }

    lazy override val toString: String = s"Matrix(rows = $rows, cols = $cols, ${arr.mkString(",")})"
  }
  object Matrix {
    def apply(rows: Int, cols: Int, nums: Double*): Matrix =
      if (nums.size != (rows * cols)) lengthError()
      else new Matrix(nums.toArray, rows, cols)
  }

  private def lengthError(): Nothing = throw new IllegalArgumentException("Lengths do not match up!")
}

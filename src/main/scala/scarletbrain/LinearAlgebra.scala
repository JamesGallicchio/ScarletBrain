package scarletbrain

object LinearAlgebra {
  case class Vector(private[LinearAlgebra] val arr: Array[Double]) {
    def +(v: Vector): Vector = if (arr.length != v.arr.length) lengthError() else {
      val other = v.arr
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) + other(i)
      Vector(a)
    }

    def +(s: Double): Vector = {
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) + s
      Vector(a)
    }

    def -(v: Vector): Vector = if (arr.length != v.arr.length) lengthError() else {
      val other = v.arr
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) - other(i)
      Vector(a)
    }

    def -(s: Double): Vector = {
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) - s
      Vector(a)
    }

    def *(v: Vector): Vector = if (arr.length != v.arr.length) lengthError() else {
      val other = v.arr
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) * other(i)
      Vector(a)
    }

    def *(s: Double): Vector = {
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) * s
      Vector(a)
    }

    def prodMatrix(v: Vector): Matrix = {
      val a = new Array[Double](arr.length * v.arr.length)
      for (i <- arr.indices) {
        val offset = i * v.arr.length
        val mult = arr(i)
        for (j <- v.arr.indices)
          a(offset + j) = mult * v.arr(j)
      }
      Matrix(a, arr.length, v.arr.length)
    }

    def map(f: Double => Double): Vector = Vector(arr.map(f))
    def sum: Double = arr.sum

    override def toString(): String =
      s"Vector(${arr.mkString(", ")})"
  }
  object Vector {
    def apply(values: Double*): Vector = new Vector(values.toArray)
  }

  case class Matrix private (private[LinearAlgebra] val arr: Array[Double], rows: Int, cols: Int) {
    def +(mx: Matrix): Matrix = if (arr.length != mx.arr.length) lengthError() else {
      val other = mx.arr
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) + other(i)
      Matrix(a, rows, cols)
    }

    def +(s: Double): Matrix = {
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) + s
      Matrix(a, rows, cols)
  }

    def -(mx: Matrix): Matrix = if (arr.length != mx.arr.length) lengthError() else {
      val other = mx.arr
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) - other(i)
      Matrix(a, rows, cols)
    }

    def rowProdSums(vec: Vector): Vector = if (vec.arr.length != cols) lengthError() else {
      val other = vec.arr
      val a = new Array[Double](rows)
      for (i <- 0 until rows) {
        val offset = i * cols
        for (j <- 0 until cols)
          a(i) += arr(offset + j) * other(j)
      }
      Vector(a)
    }

    def colProdSums(vec: Vector): Vector = if (vec.arr.length != rows) lengthError() else {
      val other = vec.arr
      val a = new Array[Double](cols)
      for (i <- 0 until rows) {
        val offset = i * cols
        for (j <- 0 until cols)
          a(j) += arr(offset + j) * other(i)
      }
      Vector(a)
    }

    def *(s: Double): Matrix = {
      val a = new Array[Double](arr.length)
      for (i <- a.indices) a(i) = arr(i) * s
      Matrix(a, rows, cols)
    }

    override def toString: String = s"Matrix(${
      arr.grouped(cols).
        map(_.mkString(",")).
        mkString("; ")
    })"
  }
  object Matrix {
    def apply(arr: Array[Double], rows: Int, cols: Int): Matrix =
      if (arr.length != (rows * cols)) lengthError()
      else new Matrix(arr, rows, cols)
  }

  private def lengthError(): Nothing = throw new IllegalArgumentException("Lengths do not match up!")
}

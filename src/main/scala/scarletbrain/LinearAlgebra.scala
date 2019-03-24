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

    def expand(newLength: Int): Vector = if (newLength < arr.length) lengthError() else {
      val a = new Array[Double](newLength)
      var i = 0
      while (i < arr.length) {
        a(i) = arr(i)
        i += 1
      }
      Vector(a)
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

  /**
    * Matrix of doubles with dimensions rows x cols, backed by a 1d array.
    * Stored row-first, e.g. for a 3x4 matrix the array indices would be:
    *     0  1  2  3
    *     4  5  6  7
    *     8  9 10 11
    *
    * @param arr array backing the matrix
    * @param rows number of rows
    * @param cols number of columns
    */
  case class Matrix private (arr: Array[Double], rows: Int, cols: Int) {

    def apply(r: Int, c: Int): Double = arr(r*cols+c)

    lazy val mag2: Double = {
      var total = 0.0
      var i = 0
      while (i < arr.length) {
        total += arr(i) * arr(i)
        i += 1
      }
      total
    }

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

    def *(s: Double): Matrix = {
      val a = new Array[Double](arr.length)
      var i = 0
      while (i < arr.length) { a(i) = arr(i) * s; i += 1 }
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

    def expand(newRows: Int, newCols: Int): Matrix = if (newRows < rows || newCols < cols) lengthError() else {
      val a = new Array[Double](newRows*newCols)
      var r = 0
      while (r < rows) {
        val offset = r*cols
        val newOffset = r*newCols
        var c = 0
        while (c < cols) {
          a(newOffset + c) = arr(offset + c)
          c += 1
        }
        r += 1
      }
      Matrix(a, newRows, newCols)
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

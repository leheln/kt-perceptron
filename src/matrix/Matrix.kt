package matrix

import java.io.Serializable
import java.lang.IndexOutOfBoundsException
import java.util.*

/**
 * A class that represents matrices.
 *
 * Most operators are overwritten to match their mathematical meaning.
 */
@Suppress("MemberVisibilityCanBePrivate")
class Matrix(val rows: Int, val columns: Int) : Serializable {
    private var data = Array(rows) { DoubleArray(columns) }

    /**
     * Secondary constructor for easy use. The [inputData] represents the content of the matrix.
     */
    constructor(rows: Int, columns: Int, vararg inputData: Double) : this(rows, columns) {
        for (inputIndex in inputData.indices) {
            val firstIndex = inputIndex / columns
            val secondIndex = inputIndex - (firstIndex * columns)
            data[firstIndex][secondIndex] = inputData[inputIndex]
        }
    }

    /**
     * Gets the Hadamard product of [member] and the matrix it gets called on
     */
    infix fun o(member: Matrix) = operatorBase(member) { left, right -> left * right }

    /**
     * Gets the sum of [member] and the matrix it gets called on
     */
    operator fun plus(member: Matrix) = operatorBase(member) { left, right -> left + right }

    /**
     * Gets the difference of [member] and the matrix it gets called on
     */
    operator fun minus(member: Matrix) = operatorBase(member) { left, right -> left - right }

    /**
     * Multiplies [member] and the matrix it gets called on
     */
    operator fun times(member: Double) = this.map { it * member }

    /**
     * Multiplies [member] and the matrix it gets called on
     *
     * If the dimensions of [member] make the multiplication impossible,
     * it throws an [IndexOutOfBoundsException]
     */
    operator fun times(member: Matrix): Matrix {
        if (this.columns != member.rows)
            throw IndexOutOfBoundsException("Incompatible dimensions for multiplying matrices: " +
                    "(${rows}, ${columns}) * (${member.rows}, ${member.columns})")

        val output = Matrix(rows, member.columns)

        for (row in data.indices) {
            for (columnMember in 0 until member.columns) {
                for (column in data[row].indices) {
                    output[row, columnMember] = output[row, columnMember] + data[row][column] * member[column, columnMember]
                }
            }
        }

        return output
    }

    /**
     * Gets the value of the matrix at a specific index through the [row, column] format
     */
    operator fun get(column: Int, row: Int) = data[column][row]

    /**
     * Sets the value of the matrix at a specific index through the [row, column] format
     */
    operator fun set(column: Int, row: Int, value: Double) {
        data[column][row] = value
    }

    /**
     * Transposes the matrix then returns it.
     */
    fun transpose(): Matrix {
        val output = Matrix(columns, rows)
        for (row in data.indices) {
            for (column in data[row].indices) {
                output[column, row] = this[row, column]
            }
        }
        return output
    }

    /**
     * Prints out the matrix in it's expected format.
     */
    fun print(): Matrix {
        for (row in data) {
            for (column in row) {
                print("%6.2f ".format(column))
            }
            println()
        }
        return map { it }
    }

    /**
     * Returns the matrix with random values between [min] and [max]
     */
    fun randomize(min: Int, max: Int): Matrix = map { Random().nextDouble() * (max - min) + min }


    /**
     * Applies the given [transform] function to each value in the matrix
     * and fills up the "output matrix" with the results.
     *
     * After that it returns the "output matrix".
     */
    fun map(transform: (Double) -> Double): Matrix {
        val output = Matrix(rows, columns)
        for (row in data.indices) {
            for (column in data[row].indices) {
                output[row, column] = transform(this[row, column])
            }
        }
        return output
    }

    /**
     * This function serves as a base for overriding most operators
     *
     * It exists to make the code cleaner and prevent code reuse
     */
    private inline fun operatorBase(member: Matrix, function: (Double, Double) -> Double): Matrix {
        if (this.rows != member.rows || this.columns != member.columns)
            throw Exception("Matrix dimensions do not match: " +
                    "(${rows}, ${columns}) * (${member.rows}, ${member.columns})")

        val output = Matrix(rows, columns)
        for (row in data.indices) {
            for (column in data[row].indices) {
                output[row, column] = function(this[row, column], member[row, column])
            }
        }
        return output
    }

}
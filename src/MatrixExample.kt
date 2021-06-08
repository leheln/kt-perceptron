import matrix.Matrix

fun main() {
    // Create a (4 x 4) Matrix filled up with zeroes
    val zeroMatrix = Matrix(4, 4)

    // Create a (3 x 4) Matrix with random numbers between -5 and 5
    val randomMatrix = Matrix(3, 4).randomize(-5, 5)

    // Matrices can be easily printed:
    println("This is my random matrix:")
    randomMatrix.print()

    // Let's create a matrix where we know its content
    // The first two variables describe the matrix's dimensions
    // The rest of the variables describe the content of it

    val exampleMatrix = Matrix(3, 5,
            1.0, 2.0, 3.4, 2.1, 3.2,
            5.2, 2.8, 1.9, 9.2, 2.1,
            9.8, 8.4, 9.6, 4.8, 3.8)

    // Now let's create two matrices to give you a few examples with the overloaded operators
    // I tried to minimize code reuse and code cleanness with the help of Kotlin's features
    // while developing those
    val a = Matrix(2, 2,
            1.0, 1.5,
            0.4, 2.0)

    val b = Matrix(2, 2,
            3.0, 6.2,
            1.8, 4.1)

    // Hadamard product (I used infix here)
    a o b

    // Matrix multiplication
    a * b
    a * 2.0

    // Matrix addition
    a + b

    // Matrix subtraction
    a - b

    // Access the values of the matrix or set them
    a[0, 0]
    a[0, 0] = 4.2

    // Now let's transpose one of the matrices:
    a.transpose()

    // And last, let's use the map function on it to duplicate its values
    a.map { it * 2 }
}
import matrix.Matrix
import neuralnetwork.NeuralNetwork
import kotlin.random.Random

fun main() {

    // Defining the dataset that the neural network will learn on:
    val matrixTrue = Matrix(1, 1,
            1.0)

    val matrixFalse = Matrix(1, 1,
            0.0)

    val matrixTrueTrue = Matrix(2, 1,
            1.0,
            1.0)

    val matrixFalseFalse = Matrix(2, 1,
            0.0,
            0.0)

    val matrixTrueFalse = Matrix(2, 1,
            1.0,
            0.0)

    val matrixFalseTrue = Matrix(2, 1,
            0.0,
            1.0)

    /*
    My goal is to teach the neural network how to solve an XOR problem
    0 means false
    1 means true

    The XOR problem is as it follows:
    (0,0) -> 0
    (1,1) -> 0
    (1,0) -> 1
    (0,1) -> 1

    It's really simple, but it's a good way to demonstrate that my API works.
    */

    // Creating a new neural network with 2 input neurons, 1 output neuron and a 0.1 learning rate
    val network = NeuralNetwork(2, 1, 0.1)

    // Let's add two layers to the network with 8 and 10 neurons respectively
    // (Really overkill for the given problem obviously.)
    network.addLayer(8)
    network.addLayer(10)

    // Compile the network before using
    network.compile()

    // Let's train the network
    for (epoch in 0..10000) {
        when (Random.nextInt(4)) {
            0 -> network.train(matrixFalseFalse, matrixFalse)
            1 -> network.train(matrixTrueTrue, matrixFalse)
            2 -> network.train(matrixTrueFalse, matrixTrue)
            3 -> network.train(matrixFalseTrue, matrixTrue)
        }
    }

    // Let's see if it worked
    println("(1,1) -> %.5f".format(network.run(matrixTrueTrue)[0, 0])) // Should be around 0.0
    println("(0,0) -> %.5f".format(network.run(matrixFalseFalse)[0, 0])) // Should be around 0.0
    println("(1,0) -> %.5f".format(network.run(matrixTrueFalse)[0, 0])) // Should be around 1.0
    println("(0,1) -> %.5f".format(network.run(matrixFalseTrue)[0, 0])) // Should be around 1.0

    // The run function returns a matrix as an output, that's why I could access it with [0, 0]

    // Obviously the API can be used in a prettier way, but I wanted the output to look good
    // So let's try it again for (0,1) which should give us a number around 1.0, this time use the built
    // in print function

    network.run(matrixFalseTrue).print()

    // Now let's save our network into a file which I'll load back later
    network.save("data.ai")

    // Let's load it back into a different object
    val newNetwork: NeuralNetwork = NeuralNetwork.loadNeuralNetwork("data.ai")

    // Let's see if it still works with the previous example
    newNetwork.run(matrixFalseTrue).print()

}
package neuralnetwork

import matrix.Matrix
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import java.io.Serializable
import kotlin.math.pow

/**
 * This class represents a neural network
 *
 * The constructor needs the number of [inputSize] and [outputSize] neurons
 * and the [learningRate] to instantiate a [NeuralNetwork] object.
 *
 * Layers can be added to the network after that. If the user is done with defining the networks structure
 * it has to be compiled before using it.
 */
@Suppress("MemberVisibilityCanBePrivate")
class NeuralNetwork(val inputSize: Int, val outputSize: Int, val learningRate: Double) : Serializable {
    private var compiled: Boolean = false
    var layers: MutableList<Matrix> = mutableListOf()
    var weights: MutableList<Matrix> = mutableListOf()
    var first: Matrix = Matrix(inputSize, 1)

    companion object {
        /**
         * Loads a neural network from a given file [path]
         */
        fun loadNeuralNetwork(path: String): NeuralNetwork {
            ObjectInputStream(FileInputStream(path)).use { it ->
                when (val network = it.readObject()) {
                    is NeuralNetwork -> return network
                    else -> throw java.lang.Exception("Deserialization failed.")
                }
            }
        }
    }

    /**
     * Saves the neural network into [path]
     */
    fun save(path: String) {
        ObjectOutputStream(FileOutputStream(path)).use { it -> it.writeObject(this) }
    }

    /**
     * Adds a new layer to the neural network. It's size is [neurons]
     */
    fun addLayer(neurons: Int) {
        if (layers.isEmpty())
            weights.add(Matrix(neurons, inputSize).randomize(-1, 1))
        else
            weights.add(Matrix(neurons, layers.last().rows).randomize(-1, 1))
        layers.add(Matrix(neurons, 1))
    }

    /**
     * Adds the output to the layer list to close up the structure of the neural network.
     * This function is needed to be called before using the neural network.
     */

    fun compile() {
        weights.add(Matrix(outputSize, layers.last().rows).randomize(-1, 1))
        layers.add(Matrix(outputSize, 1))
        compiled = true
    }

    /**
     * Feed forwards the neural network with the given [input]
     */
    fun run(input: Matrix): Matrix {
        if (!compiled) throw Exception("The neural network has to be compiled before use!")

        first = input
        if (first.rows == 1 && first.columns == inputSize)
            first = first.transpose()
        if (first.rows != inputSize || first.columns != 1)
            throw Exception("Incorrect input dimensions: (${first.rows}, ${first.columns})")

        layers[0] = singleRun(first, weights[0])
        for (layerIndex in 1 until layers.size) {
            layers[layerIndex] = singleRun(layers[layerIndex - 1], weights[layerIndex])
        }

        return layers.last()
    }

    /**
     * Helper function for [run]
     */

    private fun singleRun(input: Matrix, weights: Matrix) = (weights * input).map { sigmoid(it) }

    /**
     * Trains the neural network based on an input and a desired output
     */
    fun train(input: Matrix, target: Matrix) {
        if (!compiled) throw Exception("The neural network has to be compiled before use!")

        val output = run(input)
        val layersCopy = layers.toMutableList()
        val weightsCopy = weights.toMutableList()
        val errors: MutableList<Matrix> = mutableListOf()

        errors.add(0, target - output)
        layersCopy.removeAt(layersCopy.size - 1)

        while (layersCopy.isNotEmpty()) {
            errors.add(0, weightsCopy.last().transpose() * errors.first())
            layersCopy.removeAt(layersCopy.size - 1)
            weightsCopy.removeAt(weightsCopy.size - 1)
        }

        for (index in weights.indices) {
            if (index == 0) {
                weights[index] += (errors[index] o layers[index].map { sigmoidDerivative(it) }) * first.transpose()
            } else {
                weights[index] += (errors[index] o layers[index].map { sigmoidDerivative(it) }) * layers[index - 1].transpose()
            }

        }

    }

    /**
     * Simple sigmoid function.
     */
    fun sigmoid(x: Double): Double = 1 / (1 + Math.E.pow(-1 * x))

    /**
     * Simple function for the derivative of sigmoid
     */
    fun sigmoidDerivative(x: Double): Double = x * (1 - x)

}
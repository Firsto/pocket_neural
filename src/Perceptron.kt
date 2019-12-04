import com.opencsv.CSVReader
import com.opencsv.CSVWriter
import java.io.File
import java.io.InputStreamReader
import java.nio.file.Files
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.exp
import kotlin.math.pow


class Perceptron(
    private val inputSize: Int,
    private val outputSize: Int,
    private val hidden1Size: Int = inputSize * 5,
    private val hidden2Size: Int = outputSize * 2
) {
    lateinit var csvReader: CSVReader
    lateinit var csvWriter: CSVWriter

    val epoch = 10000
    val N = 0.1
    val M = 0.9

    val input = DoubleArray(inputSize)
    val hidden1 = DoubleArray(hidden1Size)
    val hidden2 = DoubleArray(hidden2Size)
    val output = DoubleArray(outputSize)

    val target = DoubleArray(outputSize)

    var weightsInput = Array(inputSize + 1) { initRandomDoubleArray(hidden1Size) }
    var weightsHidden = Array(hidden1Size + 1) { initRandomDoubleArray(hidden2Size) }
    var weightsOutput = Array(hidden2Size + 1) { initRandomDoubleArray(outputSize) }

    var dweightsInput = Array(inputSize + 1) { DoubleArray(hidden1Size) { 0.0 } }
    var dweightsHidden = Array(hidden1Size + 1) { DoubleArray(hidden2Size) { 0.0 } }
    var dweightsOutput = Array(hidden2Size + 1) { DoubleArray(outputSize) { 0.0 } }

    val hidden1Error = DoubleArray(hidden1Size) { 0.0 }
    val hidden2Error = DoubleArray(hidden2Size) { 0.0 }
    val outputError = DoubleArray(outputSize) { 0.0 }

    private val weightsFiles = listOf(
        Pair(File("weightsInput.csv"), weightsInput),
        Pair(File("weightsHidden.csv"), weightsHidden),
        Pair(File("weightsOutput.csv"), weightsOutput)
    )

    private fun initRandomDoubleArray(size: Int): Array<Double> {
        return Array(size) { ThreadLocalRandom.current().nextDouble(-0.3, 0.3) }
    }

    fun sigmoid(d: Double, reverse: Boolean = false): Double = if (reverse) dsigExp(d) else sigExp(d)
//    fun sigmoid(d: Double, reverse: Boolean = false): Double = if (reverse) dsigTan(d) else sigTan(d)
//
    fun sigExp(x: Double) = 1 / (1 + exp(-x))
    fun dsigExp(y: Double) = y * (1 - y)
//    fun sigTan(x: Double) = Math.tanh(x)
//    fun dsigTan(y: Double) = 1 - y * y

    init {
        for (weightsFile in weightsFiles) {
            if (!weightsFile.first.exists()) {
                weightsFile.first.createNewFile()
            }

            csvReader = CSVReader(InputStreamReader(weightsFile.first.inputStream()))
            csvReader.readAll().forEachIndexed { oi, strings ->
                strings.forEachIndexed { ii, s ->
                    weightsFile.second[oi][ii] = s.toDouble()
                }
            }
        }
    }

    fun saveWeigths() {
        for (weightsFile in weightsFiles) {
            weightsFile.first.createNewFile()

            csvWriter = CSVWriter(Files.newBufferedWriter(weightsFile.first.toPath()))

            var stringWeights: Array<String>
            weightsFile.second.forEach { weights ->
                stringWeights = Array(weights.size) { "" }
                weights.forEachIndexed { index, weight ->
                    stringWeights[index] = weight.toString()
                }
                csvWriter.writeNext(stringWeights)
                csvWriter.flushQuietly()
            }
        }
    }

    fun calculate() {
        hidden1.forEachIndexed { oi, _ ->
            var sum = weightsInput[weightsInput.size - 1][oi]
            input.forEachIndexed { ii, _ ->
                sum += input[ii] * weightsInput[ii][oi]
            }
            hidden1[oi] = sigmoid(sum)
        }
        hidden2.forEachIndexed { oi, _ ->
            var sum = weightsHidden[weightsHidden.size - 1][oi]
            hidden1.forEachIndexed { ii, _ ->
                sum += hidden1[ii] * weightsHidden[ii][oi]
            }
            hidden2[oi] = sigmoid(sum)
        }
        output.forEachIndexed { oi, _ ->
            var sum = weightsOutput[weightsOutput.size - 1][oi]
            hidden2.forEachIndexed { ii, _ ->
                sum += hidden2[ii] * weightsOutput[ii][oi]
            }
            output[oi] = sigmoid(sum)
        }
    }

    fun calculateError(truth: DoubleArray, prediction: DoubleArray): Double {

        val n = truth.size
        var rss = 0.0
        for (i in 0 until n) {
            rss += (truth[i] - prediction[i]).pow(2.0)
        }

        return Math.sqrt(rss / n)
    }

    fun train() {
        for (ep in 0 until epoch) {

            outputError.forEachIndexed { i, _ ->
                outputError[i] = (output[i] - target[i]) * sigmoid(output[i], true)
            }
            hidden2Error.forEachIndexed { oi, _ ->
                var sum = 0.0
                outputError.forEachIndexed { ii, _ ->
                    sum += outputError[ii] * weightsOutput[oi][ii] * sigmoid(hidden2[oi], true)
                }
                hidden2Error[oi] = sum
            }
            hidden1Error.forEachIndexed { oi, _ ->
                var sum = 0.0
                hidden2Error.forEachIndexed { ii, _ ->
                    sum += hidden2Error[ii] * weightsHidden[oi][ii] * sigmoid(hidden1[oi], true)
                }
                hidden1Error[oi] = sum
            }

            weightsOutput.forEachIndexed { oi, _ ->
                weightsOutput[oi].forEachIndexed { ii, _ ->
                    val inp = if (oi == hidden2.size) 1.0 else hidden2[oi]
                    dweightsOutput[oi][ii] = N * outputError[ii] * inp + M * dweightsOutput[oi][ii]
                    weightsOutput[oi][ii] += dweightsOutput[oi][ii]
                }
            }
            weightsHidden.forEachIndexed { oi, _ ->
                weightsHidden[oi].forEachIndexed { ii, _ ->
                    val inp = if (oi == hidden1.size) 1.0 else hidden1[oi]
                    dweightsHidden[oi][ii] = N * hidden2Error[ii] * inp + M * dweightsHidden[oi][ii]
                    weightsHidden[oi][ii] += dweightsHidden[oi][ii]
                }
            }
            weightsInput.forEachIndexed { oi, _ ->
                weightsInput.forEachIndexed { ii, _ ->
                    val inp = if (oi == input.size) 1.0 else input[oi]
                    dweightsInput[oi][ii] = N * hidden1Error[ii] * inp + M * dweightsInput[oi][ii]
                    weightsInput[oi][ii] += dweightsInput[oi][ii]
                }
            }

            calculate()
            if (ep % 1000 == 0) {
                println("rmse after $ep attempts = " + calculateError(target, output))
            }
        }
    }

    fun setInputs(inputArray: DoubleArray) {
        if (input.size == inputArray.size) {
            input.forEachIndexed { i, _ ->
                input[i] = inputArray[i]
            }
        }
    }

    fun setTarget(targetArray: DoubleArray) {
        if (target.size == targetArray.size) {
            target.forEachIndexed { i, _ ->
                target[i] = targetArray[i]
            }
        }
    }
}
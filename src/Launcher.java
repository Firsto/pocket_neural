import java.util.Arrays;

public class Launcher {
    public static void main(String[] args) {
        int inputSize = 2;
        int outputSize = 1;
//        Perceptron perceptron = new Perceptron(5, 2, 12, 4);
        Perceptron perceptron = new Perceptron(inputSize, outputSize, 10, 4);

//        Scanner in = new Scanner(System.in);
//        double[] input = new double[2];
//        input[0] = in.nextDouble();
//        input[1] = in.nextDouble();

        double[] input = new double[inputSize];
        input[0] = 1.0;
        input[1] = 1.0;
        perceptron.setInputs(input);
        double[] target = {1.0};
        perceptron.setTarget(target);

        logln("Inputs : " + Arrays.toString(perceptron.getInput()));

        perceptron.calculate();
        perceptron.train();
        perceptron.saveWeigths();

        showWeights(perceptron.getWeightsInput(), "input weights");
        showWeights(perceptron.getWeightsHidden(), "hidden weights");
        showWeights(perceptron.getWeightsOutput(), "output weights");
        logln("");
        double[] perceptronOutput = perceptron.getOutput();
        for (int i = 0; i < perceptronOutput.length; i++) {
            double output = perceptronOutput[i];
            logln("Output " + i + " = " + output);
        }
        logln("\n");
    }

    private static void showWeights(Double[][] weights, String name) {
        log(" -- " + name + " :");
        for (Double[] weight : weights) {
            log(" " + Arrays.toString(weight));
        }
        logln("");
    }

    private static void logln(String s) {
        System.out.println(s);
    }

    private static void log(String s) {
        System.out.print(s);
    }
}

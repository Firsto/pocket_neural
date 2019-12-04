import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Launcher {
    public static void main(String[] args) {
        int inputSize = 3;
        int outputSize = 1;
//        Perceptron perceptron = new Perceptron(5, 2, 12, 4);
        Perceptron perceptron = new Perceptron(inputSize, outputSize, 10, 4);

//        Scanner in = new Scanner(System.in);
//        double[] input = new double[2];
//        input[0] = in.nextDouble();
//        input[1] = in.nextDouble();

//        double[] input = new double[inputSize];
//        input[0] = 1.0/2;
//        input[1] = 1.0/2;
//        perceptron.setInputs(input);
//        double[] target = {1.0/4};
//        perceptron.setTarget(target);
//
//        logln("Inputs : " + Arrays.toString(perceptron.getInput()));
//
//        perceptron.calculate();
//        perceptron.train();
//        perceptron.saveWeigths();
//
//        showWeights(perceptron.getWeightsInput(), "input weights");
//        showWeights(perceptron.getWeightsHidden(), "hidden weights");
//        showWeights(perceptron.getWeightsOutput(), "output weights");
//        logln("");
//        double[] perceptronOutput = perceptron.getOutput();
//        for (int i = 0; i < perceptronOutput.length; i++) {
//            double output = perceptronOutput[i];
//            logln("Output " + i + " = " + output);
//        }
//        logln("\n");

//        double[][] inputs = {
//                {1,2},
//                {1,3},
//                {2,2},
//                {3,2},
//                {3,3},
//                {5,5},
//                {6,6},
//                {7,8},
//                {8,8},
//                {9,3},
//                {10,20},
//                {10,25},
//                {10,30},
//                {25,25},
//                {20,20},
//                {16,16},
//                {32,8},
//                {30,30},
//                {15,10},
//                {50,50}
//        };

        double[][] inputs = getRandomInputs();
        for (double[] input : inputs) {
            perceptron.setInputs(input);

            double a = input[0];
            double b = input[1];
            double ideal = a + b;

            double[] target = {ideal};
            perceptron.setTarget(target);

            perceptron.calculate();
//            perceptron.train();

            double[] perceptronOutput = perceptron.getOutput();
            for (int i = 0; i < perceptronOutput.length; i++) {
                double output = perceptronOutput[i];
                long result = Math.round(1 / output);
                String correct = (result == Math.round(ideal)) ? " is correct" : " is NOT correct";
                double error = (ideal - result) / result * 100;
                String errorString = String.format ("%.9f", error);
                logln(Math.round(a) + " + " + Math.round(b) + " = " + result + " // ideal " + Math.round(ideal) + correct + " (error " + errorString + "%)");
            }
        }
//        perceptron.saveWeigths();
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

    private static double[][] getRandomInputs() {
        double[][] inputs = new double[10][3];

        for (double[] input : inputs) {
            input[0] = Math.round(Math.random() * 1000) + 1;
            input[1] = Math.round(Math.random() * 1000) + 1;
            input[2] = 2;
        }

        return inputs;
    }
}

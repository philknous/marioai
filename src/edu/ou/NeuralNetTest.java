package edu.ou;

import java.util.Random;

public class NeuralNetTest {
	private final static int NUM_OF_INPUTS = 3;
	private final static int NUM_OF_HIDDEN = 2;
	private final static int NUM_OF_OUTPUTS = 1;
	private final static double ALPHA = 0.001;
	
	private static double avgError = 1;
	
	public static boolean TestAnd() {
		NeuralNet net = new NeuralNet(NUM_OF_INPUTS, NUM_OF_HIDDEN, NUM_OF_OUTPUTS, ALPHA);
		
		avgError = 1;
		
		double[] input = new double[] {1, 0, 0};
		double[] expectedOutput = new double[] {0};
		
		Random rand = new Random();
		double error = 0;
		
		int count = 0;
		
		while (avgError > 0.01) {
			boolean tempInput1 = rand.nextBoolean();
			boolean tempInput2 = rand.nextBoolean();
			
			boolean tempExpectedOutput = tempInput1 && tempInput2;

			input[1] = tempInput1 ? 1.0 : 0.0;
			input[2] = tempInput2 ? 1.0 : 0.0;
			
			expectedOutput[0] = tempExpectedOutput ? 1.0 : 0.0;
			
			error = net.Learn(input, expectedOutput);
						
			avgError = avgError + 0.1 * (error - avgError);
			
			if (++count % 500 == 0) {
				System.out.println(count + ": " + avgError);
			}
		}
		
		input[1] = 1.0;
		input[2] = 1.0;
		
		double[] result = net.GetOutput(input);
		
		System.out.println("1&&1:");
		System.out.println(result[0]);
		
		input[1] = 1.0;
		input[2] = 0.0;
		
		result = net.GetOutput(input);
		
		System.out.println("1&&0:");
		System.out.println(result[0]);
		
		input[1] = 0.0;
		input[2] = 1.0;
		
		result = net.GetOutput(input);
		
		System.out.println("0&&1:");
		System.out.println(result[0]);
		
		input[1] = 0.0;
		input[2] = 0.0;
		
		result = net.GetOutput(input);
		
		System.out.println("0&&0:");
		System.out.println(result[0]);
		
		
		System.out.println("SSE:");
		System.out.println(avgError);
		
		return true;
		
	}
	
	public static void TestXOR() {
		NeuralNet net = new NeuralNet(NUM_OF_INPUTS, NUM_OF_HIDDEN, NUM_OF_OUTPUTS, ALPHA);
		
		avgError = 1;

		double[] input = new double[] {1, 0, 0};
		double[] expectedOutput = new double[] {0};
		
		Random rand = new Random();
		double error = 0;
		
		int count = 0;
		
		while (avgError > 0.10) {
			boolean tempInput1 = rand.nextBoolean();
			boolean tempInput2 = rand.nextBoolean();
			
			boolean tempExpectedOutput = tempInput1 ^ tempInput2;

			input[1] = tempInput1 ? 1.0 : 0.0;
			input[2] = tempInput2 ? 1.0 : 0.0;
			
			expectedOutput[0] = tempExpectedOutput ? 1.0 : 0.0;
			
			error = net.Learn(input, expectedOutput);
						
			avgError = avgError + 0.1 * (error - avgError);
			
			if (++count % 500 == 0) {
				System.out.println(count + ": " + avgError);
			}
		}
		
		input[1] = 1.0;
		input[2] = 1.0;
		
		double[] result = net.GetOutput(input);
		
		System.out.println("1^1:");
		System.out.println(result[0]);
		
		input[1] = 1.0;
		input[2] = 0.0;
		
		result = net.GetOutput(input);
		
		System.out.println("1^0:");
		System.out.println(result[0]);
		
		input[1] = 0.0;
		input[2] = 1.0;
		
		result = net.GetOutput(input);
		
		System.out.println("0^1:");
		System.out.println(result[0]);
		
		input[1] = 0.0;
		input[2] = 0.0;
		
		result = net.GetOutput(input);
		
		System.out.println("0^0:");
		System.out.println(result[0]);
		
		
		System.out.println("SSE:");
		System.out.println(avgError);
	}
	
	public static void main(String[] args) {
		TestAnd();
		//TestXOR();
	}
}

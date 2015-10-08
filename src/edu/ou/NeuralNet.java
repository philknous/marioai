package edu.ou;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.Random;

public class NeuralNet implements Serializable {

	private static final long serialVersionUID = -615492701034743714L;
	private int numOfInputs;
	private int numOfHiddenNeurons;
	private int numOfOutputs;
	private double alpha;
	
	private double[] inputs;
	private double[][] firstLayerWeights;
	private double[][] secondLayerWeights;
	private double[] hiddenLayer;
	private double[] outputs;
	
	/**
	 * Constructor for the neural network
	 * @param numberOfInputs The number of inputs to the neural network
	 * @param numberOfHiddenNeurons The number of hidden nodes in the network
	 * @param numOfOutputs The number of outputs from the network
	 * @param alpha The learning rate when using backprop
	 */
	public NeuralNet(int numberOfInputs, int numberOfHiddenNeurons, int numOfOutputs, double alpha) {
		this.numOfInputs = numberOfInputs;
		this.numOfHiddenNeurons = numberOfHiddenNeurons;
		this.numOfOutputs = numOfOutputs;
		this.alpha = alpha;
		
		// Setup inputs, hidden neurons, and outputs
		inputs = new double[this.numOfInputs];
		hiddenLayer = new double[this.numOfHiddenNeurons];
		outputs = new double[this.numOfOutputs];
		
		// Load saved weights (if there are any)
		boolean loadedWeights = false;
		try {
			File weightFile = new File("layer1Weights.dat");
			if (weightFile.exists()) {
				FileInputStream fis = new FileInputStream(weightFile);
				ObjectInputStream ois = new ObjectInputStream(fis);
				firstLayerWeights = (double[][]) ois.readObject();
				ois.close();
				fis.close();
				
				fis = new FileInputStream("layer2Weights.dat");
				ois = new ObjectInputStream(fis);
				secondLayerWeights = (double[][]) ois.readObject();
				ois.close();
				fis.close();
				
				loadedWeights = true;
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		if (loadedWeights == false) {
			firstLayerWeights = new double[numOfInputs][numOfHiddenNeurons];
			secondLayerWeights = new double[numOfHiddenNeurons][numOfOutputs];
			
			// Initialize the weights (to random values)
			Random rand = new Random();
			
			// Input to hidden layer
			for (int from = 0; from < numOfInputs; from++) {
				for (int to = 0; to < numOfHiddenNeurons; to++) {
					firstLayerWeights[from][to] = rand.nextDouble();
				}
			}
			
			// Hidden to output layer
			for (int from = 0; from < numOfHiddenNeurons; from++) {
				for (int to = 0; to < numOfOutputs; to++) {
					secondLayerWeights[from][to] = rand.nextDouble();
				}
			}
		}
	}
	
	/**
	 * Propagates the input through the neural network and returs the output
	 * @param input The input to the network
	 * @return The output of the network given the input
	 */
	public double[] GetOutput(double[] input) {
		if (this.inputs != input) {
			System.arraycopy(input, 0, inputs, 0, input.length);
		}
		
		// Clear hidden and output layers
		for(int i = 0; i < hiddenLayer.length; i++) {
			hiddenLayer[i] = 0;
		}
		hiddenLayer[0] = 1; // Bias neuron is always 1
		
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] = 0;
		}
		
		// Send input to hidden layer
		for (int from = 0; from < inputs.length; from++) {
			for (int to = 1; to < hiddenLayer.length; to++) { // Skip the first hidden layer. It's bias
				hiddenLayer[to] += inputs[from] * firstLayerWeights[from][to];
			}
		}
		
		// Check if the hidden layer fired
		checkFire(hiddenLayer);
		
		// Send hidden layer to output
		for (int from = 0; from < hiddenLayer.length; from++) {
			for (int to = 0; to < outputs.length; to++) {
				outputs[to] += hiddenLayer[from] * secondLayerWeights[from][to];
			}
		}
		
		// Normalize the output
		double max = 0;
		for (int i = 0; i < outputs.length; i++) {
			if (outputs[i] > max) {
				max = outputs[i];
			}
		}
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] /= max;
		}
		
		
		// Check if output fired
		checkFire(outputs);
		
		return outputs;
	}
	
	/**
	 * Learns based on the given input and target output
	 * @param input The input for which to learn
	 * @param targetOutput The correct output for the given input
	 * @return The summed squared error
	 */
	public double Learn(double[] input, double[] targetOutput) {
		double[] outputError;
		double[] hiddenError;
		
		// Find what is the actual output
		this.outputs = GetOutput(input);
		
		// Find the output layer error
		outputError = getError(this.outputs, targetOutput);
		
		// Find the hidden layer error
		hiddenError = new double[hiddenLayer.length];
		
		for (int from = 0; from < hiddenLayer.length; from++) {
			double weightedError = 0;
			for (int to = 0; to < outputs.length; to++) {
				weightedError += secondLayerWeights[from][to] * outputError[to];
			}
			
			hiddenError[from] = sigmoid(hiddenLayer[from]) * weightedError;
		}
		
		// Update first layer weights
		for (int from = 0; from < inputs.length; from++) {
			for (int to = 0; to < hiddenLayer.length; to++) {
				firstLayerWeights[from][to] += alpha * hiddenError[to] * inputs[from];
			}
		}
		
		// Update second layer weights
		for (int from = 0; from < hiddenLayer.length; from++) {
			for (int to = 0; to < outputs.length; to++) {
				secondLayerWeights[from][to] += alpha * outputError[to] * hiddenLayer[from];
			}
		}
		
		double sse = 0;
		for (int i = 0; i < outputs.length; i++) {
			sse += Math.pow((targetOutput[i] - outputs[i]), 2);
		}
		
		// Average the error
		sse /= outputs.length;
		
		return sse;
	}
	
	public void Save() {
		FileOutputStream fos;
		try {
			fos = new FileOutputStream("layer1Weights.dat");
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(firstLayerWeights);
			oos.flush();
			oos.close();
			fos.flush();
			fos.close();
			
			fos = new FileOutputStream("layer2Weights.dat");
			oos = new ObjectOutputStream(fos);
			oos.writeObject(secondLayerWeights);
			oos.flush();
			oos.close();
			fos.flush();
			fos.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * Applies the sigmoid function to check if each of the neurons in a layer fired
	 * @param toCheck The layer to check
	 */
	private void checkFire(double[] toCheck){
		for (int i = 0; i < toCheck.length; i ++) {
			toCheck[i] = sigmoid(toCheck[i]);
		}
	}
	
	private double[] getError(double[] actual, double[] desired) {
		double[] error = new double[actual.length];
		
		for (int i=0; i < actual.length; i++) {
			error[i] = sigmoid(actual[i] * (desired[i] - actual[i]));
		}
		
		return error;
	}
	
	private double sigmoid(double value) {
		return 1.0d / (1.0d + Math.exp(-value));
	}
}

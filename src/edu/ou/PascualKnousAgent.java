package edu.ou;

import ch.idsia.ai.agents.ai.BasicAIAgent;
import ch.idsia.mario.environments.Environment;
import sun.rmi.runtime.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.logging.*;

public class PascualKnousAgent extends BasicAIAgent implements LearningAgent{

	private static Logger log;
	private static FileHandler fh; 
	private static SimpleFormatter formatter;
	
	private NeuralNet net;
	private int inputs = 24; // Don't forget the bias input!
	private int hiddenNeurons = 5;
	private int outputs = Environment.numberOfButtons;
	private double alpha = 0.01; // The learning rate
	
	private boolean hasLearned = false;
	private int resets = 0;
	
	private double avgError = 0;
	
	public PascualKnousAgent() {
		super("PascualKnousAgent");
		
		if (log == null) {
			
			log = Logger.getLogger(PascualKnousAgent.class.getName());
		
			try {
				log.setUseParentHandlers(false);
				fh = new FileHandler("log", 1024000, 500);
				log.addHandler(fh);
				formatter = new SimpleFormatter();
				fh.setFormatter(formatter);
			} catch (SecurityException | IOException e) {
				e.printStackTrace();
			}
		}
		
		net = new NeuralNet(inputs, hiddenNeurons, outputs, alpha);
		this.Load();
		
		reset();
	}
	
	@Override
	public void reset() {
		if (hasLearned) {
			
			resets++;
			if (resets % 50 == 0) {
				this.Save();
				resets = 0;
			}
			
			log.info("AvgErr: " + avgError);
			
			hasLearned = false;
		}
		
		//net = new NeuralNet(inputs, hiddenNeurons, outputs, alpha);
		//this.Load();
	}
	
	@Override
	public boolean[] getAction(Environment observation) {
		
		double[] inputs = buildInput(observation);
	
		double[] outputs = net.GetOutput(inputs);
		
		for (int i = 0; i < action.length; i++) {
			action[i] = outputs[i] > 0.50;
		}
		
		return action;
	}
	
	/**
	 * Learn the desired output given the environment
	 * @param observation - the current state observation
	 * @param action - the action to be learned
	 */
	public void learn(Environment observation, boolean[] action) {
		double[] inputs = buildInput(observation);
		double[] targetOutput = new double[action.length];
		
		for (int i = 0; i < targetOutput.length; i++) {
			targetOutput[i] = action[i] ? 1.0 : 0.0;
		}
		
		double error = net.Learn(inputs, targetOutput);
		
		avgError = avgError + 0.1 * (error - avgError);
		
		hasLearned = true;
	}
	
	private double[] buildInput(Environment observation) {
		byte[][] sceneObs = observation.getLevelSceneObservation();
		
		double[] inputs = new double[] {
				1,
				checkScene(sceneObs, 8, 10),
				checkScene(sceneObs, 9, 10),
				checkScene(sceneObs, 10, 10),
				checkScene(sceneObs, 11, 10),
				checkScene(sceneObs, 12, 10),
				checkScene(sceneObs, 13, 10),
				checkScene(sceneObs, 14, 10),
				checkScene(sceneObs, 8, 11),
				checkScene(sceneObs, 9, 11),
				checkScene(sceneObs, 10, 11),
				checkScene(sceneObs, 11, 11),
				checkScene(sceneObs, 12, 11),
				checkScene(sceneObs, 13, 11),
				checkScene(sceneObs, 14, 11),
				checkScene(sceneObs, 8, 12),
				checkScene(sceneObs, 9, 12),
				checkScene(sceneObs, 10, 12),
				checkScene(sceneObs, 11, 12),
				checkScene(sceneObs, 12, 12),
				checkScene(sceneObs, 13, 12),
				checkScene(sceneObs, 14, 12),
				observation.isMarioOnGround() ? 1 : 0,
				observation.mayMarioJump() ? 1 : 0};
		
		return inputs;
	}
	
	private double checkScene(byte[][] sceneObs, int x, int y) {
		return sceneObs[x][y] != 0 ? 1 : 0;
	}
	
	private void Save() {
		double[][] firstLayerWeights = net.GetWeights(1);
		double[][] secondLayerWeights = net.GetWeights(2);
		
		double[][][] temp = new double[][][] { firstLayerWeights, secondLayerWeights };
		
		FileOutputStream fos;
		try {
			fos = new FileOutputStream("brain.dat");
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(temp);
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
	
	private void Load() {
		// Load saved weights (if there are any)
		try {
			File weightFile = new File("brain.dat");
			if (weightFile.exists()) {
				double[][][] temp;
				
				FileInputStream fis = new FileInputStream(weightFile);
				ObjectInputStream ois = new ObjectInputStream(fis);
				temp = (double[][][]) ois.readObject();
				ois.close();
				fis.close();
				
				net.SetWeights(1, temp[0]);
				net.SetWeights(2, temp[1]);
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}
}

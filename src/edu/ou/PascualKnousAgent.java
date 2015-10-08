package edu.ou;

import ch.idsia.ai.agents.ai.BasicAIAgent;
import ch.idsia.mario.environments.Environment;
import sun.rmi.runtime.Log;

import java.io.IOException;
import java.util.logging.*;

public class PascualKnousAgent extends BasicAIAgent implements LearningAgent{

	private static Logger log = Logger.getLogger(PascualKnousAgent.class.getName());
	private static FileHandler fh; 
	private static SimpleFormatter formatter;
	
	private NeuralNet net;
	private int inputs = 12; // Don't forget the bias input!
	private int hiddenNeurons = 8;
	private int outputs = Environment.numberOfButtons;
	private double alpha = 0.000001; // The learning rate
	
	private boolean hasLearned = false;
	
	private int learnCount = 0;
	private double avgError = 0;
	
	public PascualKnousAgent() {
		super("PascualKnousAgent");
		try {
			log.setUseParentHandlers(false);
			fh = new FileHandler("log%u.txt", 1024000, 500);
			log.addHandler(fh);
			formatter = new SimpleFormatter();
			fh.setFormatter(formatter);
		} catch (SecurityException | IOException e) {
			e.printStackTrace();
		}
		
		reset();
	}
	
	@Override
	public void reset() {
		if (hasLearned) {
			net.Save();
			
			log.info("AvgErr: " + avgError);
			
			hasLearned = false;
		}
		
		net = new NeuralNet(inputs, hiddenNeurons, outputs, alpha);
	}
	
	@Override
	public boolean[] getAction(Environment observation) {
		
		double[] inputs = buildInput(observation);
	
		double[] outputs = net.GetOutput(inputs);
		
		for (int i = 0; i < action.length; i++) {
			action[i] = outputs[i] >= 0.70;
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
				checkScene(sceneObs, 10, 10),
				checkScene(sceneObs, 11, 10),
				checkScene(sceneObs, 12, 10),
				checkScene(sceneObs, 10, 11),
				checkScene(sceneObs, 11, 11),
				checkScene(sceneObs, 12, 11),
				checkScene(sceneObs, 10, 12),
				checkScene(sceneObs, 11, 12),
				checkScene(sceneObs, 12, 12),
				observation.isMarioOnGround() ? 1 : 0,
				observation.mayMarioJump() ? 1 : 0};
		
		return inputs;
	}
	
	private double checkScene(byte[][] sceneObs, int x, int y) {
		return sceneObs[x][y] != 0 ? 1 : 0;
	}
	
}

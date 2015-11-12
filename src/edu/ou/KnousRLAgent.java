package edu.ou;

import ch.idsia.ai.agents.ai.BasicAIAgent;
import ch.idsia.mario.engine.sprites.Mario;
import ch.idsia.mario.environments.Environment;
import edu.ou.util.SSEFormatter;
import sun.rmi.runtime.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.*;

public class KnousRLAgent extends BasicAIAgent implements LearningAgent{

	private static int JUMP_DURATION = 1;
	private static Logger log;
	private static FileHandler fh; 
	private static SSEFormatter formatter;
	
	private NeuralNet net;
	private int inputs = 973; // Don't forget the bias input!
	private int hiddenNeurons = 32;
	private int outputs = Environment.numberOfButtons;
	private double alpha = 0.05; // The learning rate
	
	private boolean hasLearned = false;
	private int resets = 0;
	
	private int jumpCount = 0;
	private int jumpHeld = 0;
	
	private double avgError = 0.3;
	
	private float prevY = 0.0f;
	private float prevX = 0.0f;
	
	private final static int MEMORY_LENGTH = 8;
	private double[][] prevInput = new double[MEMORY_LENGTH][inputs];
	private double[][] prevOutput = new double[MEMORY_LENGTH][outputs];
	private int prevStart = 0;
	private int prevEnd = 0;
	
	private boolean LearningEnabled = true;
	
	public KnousRLAgent(boolean learningEnabled) {
		super("KnousRLAgent");
		
		this.LearningEnabled = learningEnabled;
		
		if (log == null) {
			
			log = Logger.getLogger(KnousRLAgent.class.getName());
		
			try {
				log.setUseParentHandlers(false);
				fh = new FileHandler("log", 1024000, 500);
				log.addHandler(fh);
				formatter = new SSEFormatter();
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
			this.Save();
			
			log.info(avgError + "\n");
			
			hasLearned = false;
		}
		
		prevStart = 0;
		prevEnd = 0;
		
		//net = new NeuralNet(inputs, hiddenNeurons, outputs, alpha);
		//this.Load();
	}
	
	private int coins = 0;
	
	@Override
	public boolean[] getAction(Environment observation) {
		
		double[] inputs = buildInput(observation);
	
		double[] outputs = new double[action.length];
		
		if (Math.random() < 0.0) {
			for (int i = 0; i < outputs.length; i++) {
				outputs[i] = Math.random();
			}
		}
		else {
			outputs = net.GetOutput(inputs);
		}
		
		prevInput[prevEnd % MEMORY_LENGTH] = inputs;
		prevOutput[prevEnd % MEMORY_LENGTH] = outputs;
		
		prevEnd++;
		prevEnd %= MEMORY_LENGTH;
		
		if (prevEnd == prevStart) {
			prevStart++;
			prevStart %= MEMORY_LENGTH;
		}
		
		for (int i = 0; i < action.length; i++) {
			action[i] = outputs[i] > 0.50;
		}
		
		if (action[Mario.KEY_JUMP]) {
			jumpCount = JUMP_DURATION;
			jumpHeld++;
		}
		else if (jumpCount > 0 && !observation.isMarioOnGround()) {
			action[Mario.KEY_JUMP] = true;
			jumpCount--;
		}
		else if (jumpCount > 0 && observation.isMarioOnGround()){
			jumpCount = 0;
			jumpHeld = 0;
		}
		else {
			jumpHeld = 0;
		}
		
		if (jumpHeld > 12 && observation.isMarioOnGround()) {
			action[Mario.KEY_JUMP] = false;
			jumpHeld = 0;
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
	
	/**
	 * Apply the reward to the previous action
	 * @param reward - the reward for the previous action
	 */
	public void giveReward(int reward) {
		// TODO: Implement backprop for rewards
		//System.out.println("Reward: " + reward);
		if (LearningEnabled) {
			hasLearned = true;
			
			double[] randomOutput = new double[action.length];
			
			for(int i = 0; i <= Math.abs(reward); i++) {
				int temp = prevEnd;
				
				if (prevEnd < prevStart) {
					temp += MEMORY_LENGTH;
				}
				
				int power = 1;
				
				for (int hist = temp; hist >= prevStart; hist--) {
					double[] outToLearn = prevOutput[hist % MEMORY_LENGTH]; 
					
					for (int j = 0; j < randomOutput.length; j++) {
						randomOutput[j] = Math.random() > 0.5 ? 0.0 : 1.0;
						//i = Math.abs(reward);
					}
					
					//randomOutput[Mario.KEY_JUMP] = 1.0;
					//randomOutput[Mario.KEY_RIGHT] = 1.0;
					
					if (reward < 0) {
						outToLearn = randomOutput;
					}
										
					net.Learn(prevInput[hist % MEMORY_LENGTH], outToLearn, alpha / power++);
					
				}
			}
		}
	}
	
	private double[] buildInput(Environment observation) {
		byte[][] sceneObs = observation.getLevelSceneObservation();
		byte[][] enemyObs = observation.getEnemiesObservation();
		
		// Find Mario's Y Velocity
		// getMarioFloatPos() ranges from 0 to 256.0.
		// So find the difference from last frame
		// Normalize to -1.0 to 1.0
		// Cut that to -0.5 to 0.5 and shift it up by 0.5
		// So no accel is 0.5, positive is up, negative is down
		float yVel = ((prevY - observation.getMarioFloatPos()[1]) / 256 / 2) + 0.5f;
		prevY = observation.getMarioFloatPos()[1];
		
		float xVel = ((prevX - observation.getMarioFloatPos()[0]) / 256 / 2) + 0.5f;
		prevX = observation.getMarioFloatPos()[0];
		
		double[] inputs = new double[] {
				1, // Bias!
				checkScene(sceneObs, 0, 0),
				checkScene(sceneObs, 1, 0),
				checkScene(sceneObs, 2, 0),
				checkScene(sceneObs, 3, 0),
				checkScene(sceneObs, 4, 0),
				checkScene(sceneObs, 5, 0),
				checkScene(sceneObs, 6, 0),
				checkScene(sceneObs, 7, 0),
				checkScene(sceneObs, 8, 0),
				checkScene(sceneObs, 9, 0),
				checkScene(sceneObs, 10, 0),
				checkScene(sceneObs, 11, 0),
				checkScene(sceneObs, 12, 0),
				checkScene(sceneObs, 13, 0),
				checkScene(sceneObs, 14, 0),
				checkScene(sceneObs, 15, 0),
				checkScene(sceneObs, 16, 0),
				checkScene(sceneObs, 17, 0),
				checkScene(sceneObs, 18, 0),
				checkScene(sceneObs, 19, 0),
				checkScene(sceneObs, 20, 0),
				checkScene(sceneObs, 21, 0),
				checkScene(sceneObs, 0, 1),
				checkScene(sceneObs, 1, 1),
				checkScene(sceneObs, 2, 1),
				checkScene(sceneObs, 3, 1),
				checkScene(sceneObs, 4, 1),
				checkScene(sceneObs, 5, 1),
				checkScene(sceneObs, 6, 1),
				checkScene(sceneObs, 7, 1),
				checkScene(sceneObs, 8, 1),
				checkScene(sceneObs, 9, 1),
				checkScene(sceneObs, 10, 1),
				checkScene(sceneObs, 11, 1),
				checkScene(sceneObs, 12, 1),
				checkScene(sceneObs, 13, 1),
				checkScene(sceneObs, 14, 1),
				checkScene(sceneObs, 15, 1),
				checkScene(sceneObs, 16, 1),
				checkScene(sceneObs, 17, 1),
				checkScene(sceneObs, 18, 1),
				checkScene(sceneObs, 19, 1),
				checkScene(sceneObs, 20, 1),
				checkScene(sceneObs, 21, 1),
				checkScene(sceneObs, 0, 2),
				checkScene(sceneObs, 1, 2),
				checkScene(sceneObs, 2, 2),
				checkScene(sceneObs, 3, 2),
				checkScene(sceneObs, 4, 2),
				checkScene(sceneObs, 5, 2),
				checkScene(sceneObs, 6, 2),
				checkScene(sceneObs, 7, 2),
				checkScene(sceneObs, 8, 2),
				checkScene(sceneObs, 9, 2),
				checkScene(sceneObs, 10, 2),
				checkScene(sceneObs, 11, 2),
				checkScene(sceneObs, 12, 2),
				checkScene(sceneObs, 13, 2),
				checkScene(sceneObs, 14, 2),
				checkScene(sceneObs, 15, 2),
				checkScene(sceneObs, 16, 2),
				checkScene(sceneObs, 17, 2),
				checkScene(sceneObs, 18, 2),
				checkScene(sceneObs, 19, 2),
				checkScene(sceneObs, 20, 2),
				checkScene(sceneObs, 21, 2),
				checkScene(sceneObs, 0, 3),
				checkScene(sceneObs, 1, 3),
				checkScene(sceneObs, 2, 3),
				checkScene(sceneObs, 3, 3),
				checkScene(sceneObs, 4, 3),
				checkScene(sceneObs, 5, 3),
				checkScene(sceneObs, 6, 3),
				checkScene(sceneObs, 7, 3),
				checkScene(sceneObs, 8, 3),
				checkScene(sceneObs, 9, 3),
				checkScene(sceneObs, 10, 3),
				checkScene(sceneObs, 11, 3),
				checkScene(sceneObs, 12, 3),
				checkScene(sceneObs, 13, 3),
				checkScene(sceneObs, 14, 3),
				checkScene(sceneObs, 15, 3),
				checkScene(sceneObs, 16, 3),
				checkScene(sceneObs, 17, 3),
				checkScene(sceneObs, 18, 3),
				checkScene(sceneObs, 19, 3),
				checkScene(sceneObs, 20, 3),
				checkScene(sceneObs, 21, 3),
				checkScene(sceneObs, 0, 4),
				checkScene(sceneObs, 1, 4),
				checkScene(sceneObs, 2, 4),
				checkScene(sceneObs, 3, 4),
				checkScene(sceneObs, 4, 4),
				checkScene(sceneObs, 5, 4),
				checkScene(sceneObs, 6, 4),
				checkScene(sceneObs, 7, 4),
				checkScene(sceneObs, 8, 4),
				checkScene(sceneObs, 9, 4),
				checkScene(sceneObs, 10, 4),
				checkScene(sceneObs, 11, 4),
				checkScene(sceneObs, 12, 4),
				checkScene(sceneObs, 13, 4),
				checkScene(sceneObs, 14, 4),
				checkScene(sceneObs, 15, 4),
				checkScene(sceneObs, 16, 4),
				checkScene(sceneObs, 17, 4),
				checkScene(sceneObs, 18, 4),
				checkScene(sceneObs, 19, 4),
				checkScene(sceneObs, 20, 4),
				checkScene(sceneObs, 21, 4),
				checkScene(sceneObs, 0, 5),
				checkScene(sceneObs, 1, 5),
				checkScene(sceneObs, 2, 5),
				checkScene(sceneObs, 3, 5),
				checkScene(sceneObs, 4, 5),
				checkScene(sceneObs, 5, 5),
				checkScene(sceneObs, 6, 5),
				checkScene(sceneObs, 7, 5),
				checkScene(sceneObs, 8, 5),
				checkScene(sceneObs, 9, 5),
				checkScene(sceneObs, 10, 5),
				checkScene(sceneObs, 11, 5),
				checkScene(sceneObs, 12, 5),
				checkScene(sceneObs, 13, 5),
				checkScene(sceneObs, 14, 5),
				checkScene(sceneObs, 15, 5),
				checkScene(sceneObs, 16, 5),
				checkScene(sceneObs, 17, 5),
				checkScene(sceneObs, 18, 5),
				checkScene(sceneObs, 19, 5),
				checkScene(sceneObs, 20, 5),
				checkScene(sceneObs, 21, 5),
				checkScene(sceneObs, 0, 6),
				checkScene(sceneObs, 1, 6),
				checkScene(sceneObs, 2, 6),
				checkScene(sceneObs, 3, 6),
				checkScene(sceneObs, 4, 6),
				checkScene(sceneObs, 5, 6),
				checkScene(sceneObs, 6, 6),
				checkScene(sceneObs, 7, 6),
				checkScene(sceneObs, 8, 6),
				checkScene(sceneObs, 9, 6),
				checkScene(sceneObs, 10, 6),
				checkScene(sceneObs, 11, 6),
				checkScene(sceneObs, 12, 6),
				checkScene(sceneObs, 13, 6),
				checkScene(sceneObs, 14, 6),
				checkScene(sceneObs, 15, 6),
				checkScene(sceneObs, 16, 6),
				checkScene(sceneObs, 17, 6),
				checkScene(sceneObs, 18, 6),
				checkScene(sceneObs, 19, 6),
				checkScene(sceneObs, 20, 6),
				checkScene(sceneObs, 21, 6),
				checkScene(sceneObs, 0, 7),
				checkScene(sceneObs, 1, 7),
				checkScene(sceneObs, 2, 7),
				checkScene(sceneObs, 3, 7),
				checkScene(sceneObs, 4, 7),
				checkScene(sceneObs, 5, 7),
				checkScene(sceneObs, 6, 7),
				checkScene(sceneObs, 7, 7),
				checkScene(sceneObs, 8, 7),
				checkScene(sceneObs, 9, 7),
				checkScene(sceneObs, 10, 7),
				checkScene(sceneObs, 11, 7),
				checkScene(sceneObs, 12, 7),
				checkScene(sceneObs, 13, 7),
				checkScene(sceneObs, 14, 7),
				checkScene(sceneObs, 15, 7),
				checkScene(sceneObs, 16, 7),
				checkScene(sceneObs, 17, 7),
				checkScene(sceneObs, 18, 7),
				checkScene(sceneObs, 19, 7),
				checkScene(sceneObs, 20, 7),
				checkScene(sceneObs, 21, 7),
				checkScene(sceneObs, 0, 8),
				checkScene(sceneObs, 1, 8),
				checkScene(sceneObs, 2, 8),
				checkScene(sceneObs, 3, 8),
				checkScene(sceneObs, 4, 8),
				checkScene(sceneObs, 5, 8),
				checkScene(sceneObs, 6, 8),
				checkScene(sceneObs, 7, 8),
				checkScene(sceneObs, 8, 8),
				checkScene(sceneObs, 9, 8),
				checkScene(sceneObs, 10, 8),
				checkScene(sceneObs, 11, 8),
				checkScene(sceneObs, 12, 8),
				checkScene(sceneObs, 13, 8),
				checkScene(sceneObs, 14, 8),
				checkScene(sceneObs, 15, 8),
				checkScene(sceneObs, 16, 8),
				checkScene(sceneObs, 17, 8),
				checkScene(sceneObs, 18, 8),
				checkScene(sceneObs, 19, 8),
				checkScene(sceneObs, 20, 8),
				checkScene(sceneObs, 21, 8),
				checkScene(sceneObs, 0, 9),
				checkScene(sceneObs, 1, 9),
				checkScene(sceneObs, 2, 9),
				checkScene(sceneObs, 3, 9),
				checkScene(sceneObs, 4, 9),
				checkScene(sceneObs, 5, 9),
				checkScene(sceneObs, 6, 9),
				checkScene(sceneObs, 7, 9),
				checkScene(sceneObs, 8, 9),
				checkScene(sceneObs, 9, 9),
				checkScene(sceneObs, 10, 9),
				checkScene(sceneObs, 11, 9),
				checkScene(sceneObs, 12, 9),
				checkScene(sceneObs, 13, 9),
				checkScene(sceneObs, 14, 9),
				checkScene(sceneObs, 15, 9),
				checkScene(sceneObs, 16, 9),
				checkScene(sceneObs, 17, 9),
				checkScene(sceneObs, 18, 9),
				checkScene(sceneObs, 19, 9),
				checkScene(sceneObs, 20, 9),
				checkScene(sceneObs, 21, 9),
				checkScene(sceneObs, 0, 10),
				checkScene(sceneObs, 1, 10),
				checkScene(sceneObs, 2, 10),
				checkScene(sceneObs, 3, 10),
				checkScene(sceneObs, 4, 10),
				checkScene(sceneObs, 5, 10),
				checkScene(sceneObs, 6, 10),
				checkScene(sceneObs, 7, 10),
				checkScene(sceneObs, 8, 10),
				checkScene(sceneObs, 9, 10),
				checkScene(sceneObs, 10, 10),
				checkScene(sceneObs, 11, 10),
				checkScene(sceneObs, 12, 10),
				checkScene(sceneObs, 13, 10),
				checkScene(sceneObs, 14, 10),
				checkScene(sceneObs, 15, 10),
				checkScene(sceneObs, 16, 10),
				checkScene(sceneObs, 17, 10),
				checkScene(sceneObs, 18, 10),
				checkScene(sceneObs, 19, 10),
				checkScene(sceneObs, 20, 10),
				checkScene(sceneObs, 21, 10),
				checkScene(sceneObs, 0, 11),
				checkScene(sceneObs, 1, 11),
				checkScene(sceneObs, 2, 11),
				checkScene(sceneObs, 3, 11),
				checkScene(sceneObs, 4, 11),
				checkScene(sceneObs, 5, 11),
				checkScene(sceneObs, 6, 11),
				checkScene(sceneObs, 7, 11),
				checkScene(sceneObs, 8, 11),
				checkScene(sceneObs, 9, 11),
				checkScene(sceneObs, 10, 11),
				checkScene(sceneObs, 11, 11),
				checkScene(sceneObs, 12, 11),
				checkScene(sceneObs, 13, 11),
				checkScene(sceneObs, 14, 11),
				checkScene(sceneObs, 15, 11),
				checkScene(sceneObs, 16, 11),
				checkScene(sceneObs, 17, 11),
				checkScene(sceneObs, 18, 11),
				checkScene(sceneObs, 19, 11),
				checkScene(sceneObs, 20, 11),
				checkScene(sceneObs, 21, 11),
				checkScene(sceneObs, 0, 12),
				checkScene(sceneObs, 1, 12),
				checkScene(sceneObs, 2, 12),
				checkScene(sceneObs, 3, 12),
				checkScene(sceneObs, 4, 12),
				checkScene(sceneObs, 5, 12),
				checkScene(sceneObs, 6, 12),
				checkScene(sceneObs, 7, 12),
				checkScene(sceneObs, 8, 12),
				checkScene(sceneObs, 9, 12),
				checkScene(sceneObs, 10, 12),
				checkScene(sceneObs, 11, 12),
				checkScene(sceneObs, 12, 12),
				checkScene(sceneObs, 13, 12),
				checkScene(sceneObs, 14, 12),
				checkScene(sceneObs, 15, 12),
				checkScene(sceneObs, 16, 12),
				checkScene(sceneObs, 17, 12),
				checkScene(sceneObs, 18, 12),
				checkScene(sceneObs, 19, 12),
				checkScene(sceneObs, 20, 12),
				checkScene(sceneObs, 21, 12),
				checkScene(sceneObs, 0, 13),
				checkScene(sceneObs, 1, 13),
				checkScene(sceneObs, 2, 13),
				checkScene(sceneObs, 3, 13),
				checkScene(sceneObs, 4, 13),
				checkScene(sceneObs, 5, 13),
				checkScene(sceneObs, 6, 13),
				checkScene(sceneObs, 7, 13),
				checkScene(sceneObs, 8, 13),
				checkScene(sceneObs, 9, 13),
				checkScene(sceneObs, 10, 13),
				checkScene(sceneObs, 11, 13),
				checkScene(sceneObs, 12, 13),
				checkScene(sceneObs, 13, 13),
				checkScene(sceneObs, 14, 13),
				checkScene(sceneObs, 15, 13),
				checkScene(sceneObs, 16, 13),
				checkScene(sceneObs, 17, 13),
				checkScene(sceneObs, 18, 13),
				checkScene(sceneObs, 19, 13),
				checkScene(sceneObs, 20, 13),
				checkScene(sceneObs, 21, 13),
				checkScene(sceneObs, 0, 14),
				checkScene(sceneObs, 1, 14),
				checkScene(sceneObs, 2, 14),
				checkScene(sceneObs, 3, 14),
				checkScene(sceneObs, 4, 14),
				checkScene(sceneObs, 5, 14),
				checkScene(sceneObs, 6, 14),
				checkScene(sceneObs, 7, 14),
				checkScene(sceneObs, 8, 14),
				checkScene(sceneObs, 9, 14),
				checkScene(sceneObs, 10, 14),
				checkScene(sceneObs, 11, 14),
				checkScene(sceneObs, 12, 14),
				checkScene(sceneObs, 13, 14),
				checkScene(sceneObs, 14, 14),
				checkScene(sceneObs, 15, 14),
				checkScene(sceneObs, 16, 14),
				checkScene(sceneObs, 17, 14),
				checkScene(sceneObs, 18, 14),
				checkScene(sceneObs, 19, 14),
				checkScene(sceneObs, 20, 14),
				checkScene(sceneObs, 21, 14),
				checkScene(sceneObs, 0, 15),
				checkScene(sceneObs, 1, 15),
				checkScene(sceneObs, 2, 15),
				checkScene(sceneObs, 3, 15),
				checkScene(sceneObs, 4, 15),
				checkScene(sceneObs, 5, 15),
				checkScene(sceneObs, 6, 15),
				checkScene(sceneObs, 7, 15),
				checkScene(sceneObs, 8, 15),
				checkScene(sceneObs, 9, 15),
				checkScene(sceneObs, 10, 15),
				checkScene(sceneObs, 11, 15),
				checkScene(sceneObs, 12, 15),
				checkScene(sceneObs, 13, 15),
				checkScene(sceneObs, 14, 15),
				checkScene(sceneObs, 15, 15),
				checkScene(sceneObs, 16, 15),
				checkScene(sceneObs, 17, 15),
				checkScene(sceneObs, 18, 15),
				checkScene(sceneObs, 19, 15),
				checkScene(sceneObs, 20, 15),
				checkScene(sceneObs, 21, 15),
				checkScene(sceneObs, 0, 16),
				checkScene(sceneObs, 1, 16),
				checkScene(sceneObs, 2, 16),
				checkScene(sceneObs, 3, 16),
				checkScene(sceneObs, 4, 16),
				checkScene(sceneObs, 5, 16),
				checkScene(sceneObs, 6, 16),
				checkScene(sceneObs, 7, 16),
				checkScene(sceneObs, 8, 16),
				checkScene(sceneObs, 9, 16),
				checkScene(sceneObs, 10, 16),
				checkScene(sceneObs, 11, 16),
				checkScene(sceneObs, 12, 16),
				checkScene(sceneObs, 13, 16),
				checkScene(sceneObs, 14, 16),
				checkScene(sceneObs, 15, 16),
				checkScene(sceneObs, 16, 16),
				checkScene(sceneObs, 17, 16),
				checkScene(sceneObs, 18, 16),
				checkScene(sceneObs, 19, 16),
				checkScene(sceneObs, 20, 16),
				checkScene(sceneObs, 21, 16),
				checkScene(sceneObs, 0, 17),
				checkScene(sceneObs, 1, 17),
				checkScene(sceneObs, 2, 17),
				checkScene(sceneObs, 3, 17),
				checkScene(sceneObs, 4, 17),
				checkScene(sceneObs, 5, 17),
				checkScene(sceneObs, 6, 17),
				checkScene(sceneObs, 7, 17),
				checkScene(sceneObs, 8, 17),
				checkScene(sceneObs, 9, 17),
				checkScene(sceneObs, 10, 17),
				checkScene(sceneObs, 11, 17),
				checkScene(sceneObs, 12, 17),
				checkScene(sceneObs, 13, 17),
				checkScene(sceneObs, 14, 17),
				checkScene(sceneObs, 15, 17),
				checkScene(sceneObs, 16, 17),
				checkScene(sceneObs, 17, 17),
				checkScene(sceneObs, 18, 17),
				checkScene(sceneObs, 19, 17),
				checkScene(sceneObs, 20, 17),
				checkScene(sceneObs, 21, 17),
				checkScene(sceneObs, 0, 18),
				checkScene(sceneObs, 1, 18),
				checkScene(sceneObs, 2, 18),
				checkScene(sceneObs, 3, 18),
				checkScene(sceneObs, 4, 18),
				checkScene(sceneObs, 5, 18),
				checkScene(sceneObs, 6, 18),
				checkScene(sceneObs, 7, 18),
				checkScene(sceneObs, 8, 18),
				checkScene(sceneObs, 9, 18),
				checkScene(sceneObs, 10, 18),
				checkScene(sceneObs, 11, 18),
				checkScene(sceneObs, 12, 18),
				checkScene(sceneObs, 13, 18),
				checkScene(sceneObs, 14, 18),
				checkScene(sceneObs, 15, 18),
				checkScene(sceneObs, 16, 18),
				checkScene(sceneObs, 17, 18),
				checkScene(sceneObs, 18, 18),
				checkScene(sceneObs, 19, 18),
				checkScene(sceneObs, 20, 18),
				checkScene(sceneObs, 21, 18),
				checkScene(sceneObs, 0, 19),
				checkScene(sceneObs, 1, 19),
				checkScene(sceneObs, 2, 19),
				checkScene(sceneObs, 3, 19),
				checkScene(sceneObs, 4, 19),
				checkScene(sceneObs, 5, 19),
				checkScene(sceneObs, 6, 19),
				checkScene(sceneObs, 7, 19),
				checkScene(sceneObs, 8, 19),
				checkScene(sceneObs, 9, 19),
				checkScene(sceneObs, 10, 19),
				checkScene(sceneObs, 11, 19),
				checkScene(sceneObs, 12, 19),
				checkScene(sceneObs, 13, 19),
				checkScene(sceneObs, 14, 19),
				checkScene(sceneObs, 15, 19),
				checkScene(sceneObs, 16, 19),
				checkScene(sceneObs, 17, 19),
				checkScene(sceneObs, 18, 19),
				checkScene(sceneObs, 19, 19),
				checkScene(sceneObs, 20, 19),
				checkScene(sceneObs, 21, 19),
				checkScene(sceneObs, 0, 20),
				checkScene(sceneObs, 1, 20),
				checkScene(sceneObs, 2, 20),
				checkScene(sceneObs, 3, 20),
				checkScene(sceneObs, 4, 20),
				checkScene(sceneObs, 5, 20),
				checkScene(sceneObs, 6, 20),
				checkScene(sceneObs, 7, 20),
				checkScene(sceneObs, 8, 20),
				checkScene(sceneObs, 9, 20),
				checkScene(sceneObs, 10, 20),
				checkScene(sceneObs, 11, 20),
				checkScene(sceneObs, 12, 20),
				checkScene(sceneObs, 13, 20),
				checkScene(sceneObs, 14, 20),
				checkScene(sceneObs, 15, 20),
				checkScene(sceneObs, 16, 20),
				checkScene(sceneObs, 17, 20),
				checkScene(sceneObs, 18, 20),
				checkScene(sceneObs, 19, 20),
				checkScene(sceneObs, 20, 20),
				checkScene(sceneObs, 21, 20),
				checkScene(sceneObs, 0, 21),
				checkScene(sceneObs, 1, 21),
				checkScene(sceneObs, 2, 21),
				checkScene(sceneObs, 3, 21),
				checkScene(sceneObs, 4, 21),
				checkScene(sceneObs, 5, 21),
				checkScene(sceneObs, 6, 21),
				checkScene(sceneObs, 7, 21),
				checkScene(sceneObs, 8, 21),
				checkScene(sceneObs, 9, 21),
				checkScene(sceneObs, 10, 21),
				checkScene(sceneObs, 11, 21),
				checkScene(sceneObs, 12, 21),
				checkScene(sceneObs, 13, 21),
				checkScene(sceneObs, 14, 21),
				checkScene(sceneObs, 15, 21),
				checkScene(sceneObs, 16, 21),
				checkScene(sceneObs, 17, 21),
				checkScene(sceneObs, 18, 21),
				checkScene(sceneObs, 19, 21),
				checkScene(sceneObs, 20, 21),
				checkScene(sceneObs, 21, 21),
				checkForEnemy(enemyObs, 0, 0),
				checkForEnemy(enemyObs, 1, 0),
				checkForEnemy(enemyObs, 2, 0),
				checkForEnemy(enemyObs, 3, 0),
				checkForEnemy(enemyObs, 4, 0),
				checkForEnemy(enemyObs, 5, 0),
				checkForEnemy(enemyObs, 6, 0),
				checkForEnemy(enemyObs, 7, 0),
				checkForEnemy(enemyObs, 8, 0),
				checkForEnemy(enemyObs, 9, 0),
				checkForEnemy(enemyObs, 10, 0),
				checkForEnemy(enemyObs, 11, 0),
				checkForEnemy(enemyObs, 12, 0),
				checkForEnemy(enemyObs, 13, 0),
				checkForEnemy(enemyObs, 14, 0),
				checkForEnemy(enemyObs, 15, 0),
				checkForEnemy(enemyObs, 16, 0),
				checkForEnemy(enemyObs, 17, 0),
				checkForEnemy(enemyObs, 18, 0),
				checkForEnemy(enemyObs, 19, 0),
				checkForEnemy(enemyObs, 20, 0),
				checkForEnemy(enemyObs, 21, 0),
				checkForEnemy(enemyObs, 0, 1),
				checkForEnemy(enemyObs, 1, 1),
				checkForEnemy(enemyObs, 2, 1),
				checkForEnemy(enemyObs, 3, 1),
				checkForEnemy(enemyObs, 4, 1),
				checkForEnemy(enemyObs, 5, 1),
				checkForEnemy(enemyObs, 6, 1),
				checkForEnemy(enemyObs, 7, 1),
				checkForEnemy(enemyObs, 8, 1),
				checkForEnemy(enemyObs, 9, 1),
				checkForEnemy(enemyObs, 10, 1),
				checkForEnemy(enemyObs, 11, 1),
				checkForEnemy(enemyObs, 12, 1),
				checkForEnemy(enemyObs, 13, 1),
				checkForEnemy(enemyObs, 14, 1),
				checkForEnemy(enemyObs, 15, 1),
				checkForEnemy(enemyObs, 16, 1),
				checkForEnemy(enemyObs, 17, 1),
				checkForEnemy(enemyObs, 18, 1),
				checkForEnemy(enemyObs, 19, 1),
				checkForEnemy(enemyObs, 20, 1),
				checkForEnemy(enemyObs, 21, 1),
				checkForEnemy(enemyObs, 0, 2),
				checkForEnemy(enemyObs, 1, 2),
				checkForEnemy(enemyObs, 2, 2),
				checkForEnemy(enemyObs, 3, 2),
				checkForEnemy(enemyObs, 4, 2),
				checkForEnemy(enemyObs, 5, 2),
				checkForEnemy(enemyObs, 6, 2),
				checkForEnemy(enemyObs, 7, 2),
				checkForEnemy(enemyObs, 8, 2),
				checkForEnemy(enemyObs, 9, 2),
				checkForEnemy(enemyObs, 10, 2),
				checkForEnemy(enemyObs, 11, 2),
				checkForEnemy(enemyObs, 12, 2),
				checkForEnemy(enemyObs, 13, 2),
				checkForEnemy(enemyObs, 14, 2),
				checkForEnemy(enemyObs, 15, 2),
				checkForEnemy(enemyObs, 16, 2),
				checkForEnemy(enemyObs, 17, 2),
				checkForEnemy(enemyObs, 18, 2),
				checkForEnemy(enemyObs, 19, 2),
				checkForEnemy(enemyObs, 20, 2),
				checkForEnemy(enemyObs, 21, 2),
				checkForEnemy(enemyObs, 0, 3),
				checkForEnemy(enemyObs, 1, 3),
				checkForEnemy(enemyObs, 2, 3),
				checkForEnemy(enemyObs, 3, 3),
				checkForEnemy(enemyObs, 4, 3),
				checkForEnemy(enemyObs, 5, 3),
				checkForEnemy(enemyObs, 6, 3),
				checkForEnemy(enemyObs, 7, 3),
				checkForEnemy(enemyObs, 8, 3),
				checkForEnemy(enemyObs, 9, 3),
				checkForEnemy(enemyObs, 10, 3),
				checkForEnemy(enemyObs, 11, 3),
				checkForEnemy(enemyObs, 12, 3),
				checkForEnemy(enemyObs, 13, 3),
				checkForEnemy(enemyObs, 14, 3),
				checkForEnemy(enemyObs, 15, 3),
				checkForEnemy(enemyObs, 16, 3),
				checkForEnemy(enemyObs, 17, 3),
				checkForEnemy(enemyObs, 18, 3),
				checkForEnemy(enemyObs, 19, 3),
				checkForEnemy(enemyObs, 20, 3),
				checkForEnemy(enemyObs, 21, 3),
				checkForEnemy(enemyObs, 0, 4),
				checkForEnemy(enemyObs, 1, 4),
				checkForEnemy(enemyObs, 2, 4),
				checkForEnemy(enemyObs, 3, 4),
				checkForEnemy(enemyObs, 4, 4),
				checkForEnemy(enemyObs, 5, 4),
				checkForEnemy(enemyObs, 6, 4),
				checkForEnemy(enemyObs, 7, 4),
				checkForEnemy(enemyObs, 8, 4),
				checkForEnemy(enemyObs, 9, 4),
				checkForEnemy(enemyObs, 10, 4),
				checkForEnemy(enemyObs, 11, 4),
				checkForEnemy(enemyObs, 12, 4),
				checkForEnemy(enemyObs, 13, 4),
				checkForEnemy(enemyObs, 14, 4),
				checkForEnemy(enemyObs, 15, 4),
				checkForEnemy(enemyObs, 16, 4),
				checkForEnemy(enemyObs, 17, 4),
				checkForEnemy(enemyObs, 18, 4),
				checkForEnemy(enemyObs, 19, 4),
				checkForEnemy(enemyObs, 20, 4),
				checkForEnemy(enemyObs, 21, 4),
				checkForEnemy(enemyObs, 0, 5),
				checkForEnemy(enemyObs, 1, 5),
				checkForEnemy(enemyObs, 2, 5),
				checkForEnemy(enemyObs, 3, 5),
				checkForEnemy(enemyObs, 4, 5),
				checkForEnemy(enemyObs, 5, 5),
				checkForEnemy(enemyObs, 6, 5),
				checkForEnemy(enemyObs, 7, 5),
				checkForEnemy(enemyObs, 8, 5),
				checkForEnemy(enemyObs, 9, 5),
				checkForEnemy(enemyObs, 10, 5),
				checkForEnemy(enemyObs, 11, 5),
				checkForEnemy(enemyObs, 12, 5),
				checkForEnemy(enemyObs, 13, 5),
				checkForEnemy(enemyObs, 14, 5),
				checkForEnemy(enemyObs, 15, 5),
				checkForEnemy(enemyObs, 16, 5),
				checkForEnemy(enemyObs, 17, 5),
				checkForEnemy(enemyObs, 18, 5),
				checkForEnemy(enemyObs, 19, 5),
				checkForEnemy(enemyObs, 20, 5),
				checkForEnemy(enemyObs, 21, 5),
				checkForEnemy(enemyObs, 0, 6),
				checkForEnemy(enemyObs, 1, 6),
				checkForEnemy(enemyObs, 2, 6),
				checkForEnemy(enemyObs, 3, 6),
				checkForEnemy(enemyObs, 4, 6),
				checkForEnemy(enemyObs, 5, 6),
				checkForEnemy(enemyObs, 6, 6),
				checkForEnemy(enemyObs, 7, 6),
				checkForEnemy(enemyObs, 8, 6),
				checkForEnemy(enemyObs, 9, 6),
				checkForEnemy(enemyObs, 10, 6),
				checkForEnemy(enemyObs, 11, 6),
				checkForEnemy(enemyObs, 12, 6),
				checkForEnemy(enemyObs, 13, 6),
				checkForEnemy(enemyObs, 14, 6),
				checkForEnemy(enemyObs, 15, 6),
				checkForEnemy(enemyObs, 16, 6),
				checkForEnemy(enemyObs, 17, 6),
				checkForEnemy(enemyObs, 18, 6),
				checkForEnemy(enemyObs, 19, 6),
				checkForEnemy(enemyObs, 20, 6),
				checkForEnemy(enemyObs, 21, 6),
				checkForEnemy(enemyObs, 0, 7),
				checkForEnemy(enemyObs, 1, 7),
				checkForEnemy(enemyObs, 2, 7),
				checkForEnemy(enemyObs, 3, 7),
				checkForEnemy(enemyObs, 4, 7),
				checkForEnemy(enemyObs, 5, 7),
				checkForEnemy(enemyObs, 6, 7),
				checkForEnemy(enemyObs, 7, 7),
				checkForEnemy(enemyObs, 8, 7),
				checkForEnemy(enemyObs, 9, 7),
				checkForEnemy(enemyObs, 10, 7),
				checkForEnemy(enemyObs, 11, 7),
				checkForEnemy(enemyObs, 12, 7),
				checkForEnemy(enemyObs, 13, 7),
				checkForEnemy(enemyObs, 14, 7),
				checkForEnemy(enemyObs, 15, 7),
				checkForEnemy(enemyObs, 16, 7),
				checkForEnemy(enemyObs, 17, 7),
				checkForEnemy(enemyObs, 18, 7),
				checkForEnemy(enemyObs, 19, 7),
				checkForEnemy(enemyObs, 20, 7),
				checkForEnemy(enemyObs, 21, 7),
				checkForEnemy(enemyObs, 0, 8),
				checkForEnemy(enemyObs, 1, 8),
				checkForEnemy(enemyObs, 2, 8),
				checkForEnemy(enemyObs, 3, 8),
				checkForEnemy(enemyObs, 4, 8),
				checkForEnemy(enemyObs, 5, 8),
				checkForEnemy(enemyObs, 6, 8),
				checkForEnemy(enemyObs, 7, 8),
				checkForEnemy(enemyObs, 8, 8),
				checkForEnemy(enemyObs, 9, 8),
				checkForEnemy(enemyObs, 10, 8),
				checkForEnemy(enemyObs, 11, 8),
				checkForEnemy(enemyObs, 12, 8),
				checkForEnemy(enemyObs, 13, 8),
				checkForEnemy(enemyObs, 14, 8),
				checkForEnemy(enemyObs, 15, 8),
				checkForEnemy(enemyObs, 16, 8),
				checkForEnemy(enemyObs, 17, 8),
				checkForEnemy(enemyObs, 18, 8),
				checkForEnemy(enemyObs, 19, 8),
				checkForEnemy(enemyObs, 20, 8),
				checkForEnemy(enemyObs, 21, 8),
				checkForEnemy(enemyObs, 0, 9),
				checkForEnemy(enemyObs, 1, 9),
				checkForEnemy(enemyObs, 2, 9),
				checkForEnemy(enemyObs, 3, 9),
				checkForEnemy(enemyObs, 4, 9),
				checkForEnemy(enemyObs, 5, 9),
				checkForEnemy(enemyObs, 6, 9),
				checkForEnemy(enemyObs, 7, 9),
				checkForEnemy(enemyObs, 8, 9),
				checkForEnemy(enemyObs, 9, 9),
				checkForEnemy(enemyObs, 10, 9),
				checkForEnemy(enemyObs, 11, 9),
				checkForEnemy(enemyObs, 12, 9),
				checkForEnemy(enemyObs, 13, 9),
				checkForEnemy(enemyObs, 14, 9),
				checkForEnemy(enemyObs, 15, 9),
				checkForEnemy(enemyObs, 16, 9),
				checkForEnemy(enemyObs, 17, 9),
				checkForEnemy(enemyObs, 18, 9),
				checkForEnemy(enemyObs, 19, 9),
				checkForEnemy(enemyObs, 20, 9),
				checkForEnemy(enemyObs, 21, 9),
				checkForEnemy(enemyObs, 0, 10),
				checkForEnemy(enemyObs, 1, 10),
				checkForEnemy(enemyObs, 2, 10),
				checkForEnemy(enemyObs, 3, 10),
				checkForEnemy(enemyObs, 4, 10),
				checkForEnemy(enemyObs, 5, 10),
				checkForEnemy(enemyObs, 6, 10),
				checkForEnemy(enemyObs, 7, 10),
				checkForEnemy(enemyObs, 8, 10),
				checkForEnemy(enemyObs, 9, 10),
				checkForEnemy(enemyObs, 10, 10),
				checkForEnemy(enemyObs, 11, 10),
				checkForEnemy(enemyObs, 12, 10),
				checkForEnemy(enemyObs, 13, 10),
				checkForEnemy(enemyObs, 14, 10),
				checkForEnemy(enemyObs, 15, 10),
				checkForEnemy(enemyObs, 16, 10),
				checkForEnemy(enemyObs, 17, 10),
				checkForEnemy(enemyObs, 18, 10),
				checkForEnemy(enemyObs, 19, 10),
				checkForEnemy(enemyObs, 20, 10),
				checkForEnemy(enemyObs, 21, 10),
				checkForEnemy(enemyObs, 0, 11),
				checkForEnemy(enemyObs, 1, 11),
				checkForEnemy(enemyObs, 2, 11),
				checkForEnemy(enemyObs, 3, 11),
				checkForEnemy(enemyObs, 4, 11),
				checkForEnemy(enemyObs, 5, 11),
				checkForEnemy(enemyObs, 6, 11),
				checkForEnemy(enemyObs, 7, 11),
				checkForEnemy(enemyObs, 8, 11),
				checkForEnemy(enemyObs, 9, 11),
				checkForEnemy(enemyObs, 10, 11),
				checkForEnemy(enemyObs, 11, 11),
				checkForEnemy(enemyObs, 12, 11),
				checkForEnemy(enemyObs, 13, 11),
				checkForEnemy(enemyObs, 14, 11),
				checkForEnemy(enemyObs, 15, 11),
				checkForEnemy(enemyObs, 16, 11),
				checkForEnemy(enemyObs, 17, 11),
				checkForEnemy(enemyObs, 18, 11),
				checkForEnemy(enemyObs, 19, 11),
				checkForEnemy(enemyObs, 20, 11),
				checkForEnemy(enemyObs, 21, 11),
				checkForEnemy(enemyObs, 0, 12),
				checkForEnemy(enemyObs, 1, 12),
				checkForEnemy(enemyObs, 2, 12),
				checkForEnemy(enemyObs, 3, 12),
				checkForEnemy(enemyObs, 4, 12),
				checkForEnemy(enemyObs, 5, 12),
				checkForEnemy(enemyObs, 6, 12),
				checkForEnemy(enemyObs, 7, 12),
				checkForEnemy(enemyObs, 8, 12),
				checkForEnemy(enemyObs, 9, 12),
				checkForEnemy(enemyObs, 10, 12),
				checkForEnemy(enemyObs, 11, 12),
				checkForEnemy(enemyObs, 12, 12),
				checkForEnemy(enemyObs, 13, 12),
				checkForEnemy(enemyObs, 14, 12),
				checkForEnemy(enemyObs, 15, 12),
				checkForEnemy(enemyObs, 16, 12),
				checkForEnemy(enemyObs, 17, 12),
				checkForEnemy(enemyObs, 18, 12),
				checkForEnemy(enemyObs, 19, 12),
				checkForEnemy(enemyObs, 20, 12),
				checkForEnemy(enemyObs, 21, 12),
				checkForEnemy(enemyObs, 0, 13),
				checkForEnemy(enemyObs, 1, 13),
				checkForEnemy(enemyObs, 2, 13),
				checkForEnemy(enemyObs, 3, 13),
				checkForEnemy(enemyObs, 4, 13),
				checkForEnemy(enemyObs, 5, 13),
				checkForEnemy(enemyObs, 6, 13),
				checkForEnemy(enemyObs, 7, 13),
				checkForEnemy(enemyObs, 8, 13),
				checkForEnemy(enemyObs, 9, 13),
				checkForEnemy(enemyObs, 10, 13),
				checkForEnemy(enemyObs, 11, 13),
				checkForEnemy(enemyObs, 12, 13),
				checkForEnemy(enemyObs, 13, 13),
				checkForEnemy(enemyObs, 14, 13),
				checkForEnemy(enemyObs, 15, 13),
				checkForEnemy(enemyObs, 16, 13),
				checkForEnemy(enemyObs, 17, 13),
				checkForEnemy(enemyObs, 18, 13),
				checkForEnemy(enemyObs, 19, 13),
				checkForEnemy(enemyObs, 20, 13),
				checkForEnemy(enemyObs, 21, 13),
				checkForEnemy(enemyObs, 0, 14),
				checkForEnemy(enemyObs, 1, 14),
				checkForEnemy(enemyObs, 2, 14),
				checkForEnemy(enemyObs, 3, 14),
				checkForEnemy(enemyObs, 4, 14),
				checkForEnemy(enemyObs, 5, 14),
				checkForEnemy(enemyObs, 6, 14),
				checkForEnemy(enemyObs, 7, 14),
				checkForEnemy(enemyObs, 8, 14),
				checkForEnemy(enemyObs, 9, 14),
				checkForEnemy(enemyObs, 10, 14),
				checkForEnemy(enemyObs, 11, 14),
				checkForEnemy(enemyObs, 12, 14),
				checkForEnemy(enemyObs, 13, 14),
				checkForEnemy(enemyObs, 14, 14),
				checkForEnemy(enemyObs, 15, 14),
				checkForEnemy(enemyObs, 16, 14),
				checkForEnemy(enemyObs, 17, 14),
				checkForEnemy(enemyObs, 18, 14),
				checkForEnemy(enemyObs, 19, 14),
				checkForEnemy(enemyObs, 20, 14),
				checkForEnemy(enemyObs, 21, 14),
				checkForEnemy(enemyObs, 0, 15),
				checkForEnemy(enemyObs, 1, 15),
				checkForEnemy(enemyObs, 2, 15),
				checkForEnemy(enemyObs, 3, 15),
				checkForEnemy(enemyObs, 4, 15),
				checkForEnemy(enemyObs, 5, 15),
				checkForEnemy(enemyObs, 6, 15),
				checkForEnemy(enemyObs, 7, 15),
				checkForEnemy(enemyObs, 8, 15),
				checkForEnemy(enemyObs, 9, 15),
				checkForEnemy(enemyObs, 10, 15),
				checkForEnemy(enemyObs, 11, 15),
				checkForEnemy(enemyObs, 12, 15),
				checkForEnemy(enemyObs, 13, 15),
				checkForEnemy(enemyObs, 14, 15),
				checkForEnemy(enemyObs, 15, 15),
				checkForEnemy(enemyObs, 16, 15),
				checkForEnemy(enemyObs, 17, 15),
				checkForEnemy(enemyObs, 18, 15),
				checkForEnemy(enemyObs, 19, 15),
				checkForEnemy(enemyObs, 20, 15),
				checkForEnemy(enemyObs, 21, 15),
				checkForEnemy(enemyObs, 0, 16),
				checkForEnemy(enemyObs, 1, 16),
				checkForEnemy(enemyObs, 2, 16),
				checkForEnemy(enemyObs, 3, 16),
				checkForEnemy(enemyObs, 4, 16),
				checkForEnemy(enemyObs, 5, 16),
				checkForEnemy(enemyObs, 6, 16),
				checkForEnemy(enemyObs, 7, 16),
				checkForEnemy(enemyObs, 8, 16),
				checkForEnemy(enemyObs, 9, 16),
				checkForEnemy(enemyObs, 10, 16),
				checkForEnemy(enemyObs, 11, 16),
				checkForEnemy(enemyObs, 12, 16),
				checkForEnemy(enemyObs, 13, 16),
				checkForEnemy(enemyObs, 14, 16),
				checkForEnemy(enemyObs, 15, 16),
				checkForEnemy(enemyObs, 16, 16),
				checkForEnemy(enemyObs, 17, 16),
				checkForEnemy(enemyObs, 18, 16),
				checkForEnemy(enemyObs, 19, 16),
				checkForEnemy(enemyObs, 20, 16),
				checkForEnemy(enemyObs, 21, 16),
				checkForEnemy(enemyObs, 0, 17),
				checkForEnemy(enemyObs, 1, 17),
				checkForEnemy(enemyObs, 2, 17),
				checkForEnemy(enemyObs, 3, 17),
				checkForEnemy(enemyObs, 4, 17),
				checkForEnemy(enemyObs, 5, 17),
				checkForEnemy(enemyObs, 6, 17),
				checkForEnemy(enemyObs, 7, 17),
				checkForEnemy(enemyObs, 8, 17),
				checkForEnemy(enemyObs, 9, 17),
				checkForEnemy(enemyObs, 10, 17),
				checkForEnemy(enemyObs, 11, 17),
				checkForEnemy(enemyObs, 12, 17),
				checkForEnemy(enemyObs, 13, 17),
				checkForEnemy(enemyObs, 14, 17),
				checkForEnemy(enemyObs, 15, 17),
				checkForEnemy(enemyObs, 16, 17),
				checkForEnemy(enemyObs, 17, 17),
				checkForEnemy(enemyObs, 18, 17),
				checkForEnemy(enemyObs, 19, 17),
				checkForEnemy(enemyObs, 20, 17),
				checkForEnemy(enemyObs, 21, 17),
				checkForEnemy(enemyObs, 0, 18),
				checkForEnemy(enemyObs, 1, 18),
				checkForEnemy(enemyObs, 2, 18),
				checkForEnemy(enemyObs, 3, 18),
				checkForEnemy(enemyObs, 4, 18),
				checkForEnemy(enemyObs, 5, 18),
				checkForEnemy(enemyObs, 6, 18),
				checkForEnemy(enemyObs, 7, 18),
				checkForEnemy(enemyObs, 8, 18),
				checkForEnemy(enemyObs, 9, 18),
				checkForEnemy(enemyObs, 10, 18),
				checkForEnemy(enemyObs, 11, 18),
				checkForEnemy(enemyObs, 12, 18),
				checkForEnemy(enemyObs, 13, 18),
				checkForEnemy(enemyObs, 14, 18),
				checkForEnemy(enemyObs, 15, 18),
				checkForEnemy(enemyObs, 16, 18),
				checkForEnemy(enemyObs, 17, 18),
				checkForEnemy(enemyObs, 18, 18),
				checkForEnemy(enemyObs, 19, 18),
				checkForEnemy(enemyObs, 20, 18),
				checkForEnemy(enemyObs, 21, 18),
				checkForEnemy(enemyObs, 0, 19),
				checkForEnemy(enemyObs, 1, 19),
				checkForEnemy(enemyObs, 2, 19),
				checkForEnemy(enemyObs, 3, 19),
				checkForEnemy(enemyObs, 4, 19),
				checkForEnemy(enemyObs, 5, 19),
				checkForEnemy(enemyObs, 6, 19),
				checkForEnemy(enemyObs, 7, 19),
				checkForEnemy(enemyObs, 8, 19),
				checkForEnemy(enemyObs, 9, 19),
				checkForEnemy(enemyObs, 10, 19),
				checkForEnemy(enemyObs, 11, 19),
				checkForEnemy(enemyObs, 12, 19),
				checkForEnemy(enemyObs, 13, 19),
				checkForEnemy(enemyObs, 14, 19),
				checkForEnemy(enemyObs, 15, 19),
				checkForEnemy(enemyObs, 16, 19),
				checkForEnemy(enemyObs, 17, 19),
				checkForEnemy(enemyObs, 18, 19),
				checkForEnemy(enemyObs, 19, 19),
				checkForEnemy(enemyObs, 20, 19),
				checkForEnemy(enemyObs, 21, 19),
				checkForEnemy(enemyObs, 0, 20),
				checkForEnemy(enemyObs, 1, 20),
				checkForEnemy(enemyObs, 2, 20),
				checkForEnemy(enemyObs, 3, 20),
				checkForEnemy(enemyObs, 4, 20),
				checkForEnemy(enemyObs, 5, 20),
				checkForEnemy(enemyObs, 6, 20),
				checkForEnemy(enemyObs, 7, 20),
				checkForEnemy(enemyObs, 8, 20),
				checkForEnemy(enemyObs, 9, 20),
				checkForEnemy(enemyObs, 10, 20),
				checkForEnemy(enemyObs, 11, 20),
				checkForEnemy(enemyObs, 12, 20),
				checkForEnemy(enemyObs, 13, 20),
				checkForEnemy(enemyObs, 14, 20),
				checkForEnemy(enemyObs, 15, 20),
				checkForEnemy(enemyObs, 16, 20),
				checkForEnemy(enemyObs, 17, 20),
				checkForEnemy(enemyObs, 18, 20),
				checkForEnemy(enemyObs, 19, 20),
				checkForEnemy(enemyObs, 20, 20),
				checkForEnemy(enemyObs, 21, 20),
				checkForEnemy(enemyObs, 0, 21),
				checkForEnemy(enemyObs, 1, 21),
				checkForEnemy(enemyObs, 2, 21),
				checkForEnemy(enemyObs, 3, 21),
				checkForEnemy(enemyObs, 4, 21),
				checkForEnemy(enemyObs, 5, 21),
				checkForEnemy(enemyObs, 6, 21),
				checkForEnemy(enemyObs, 7, 21),
				checkForEnemy(enemyObs, 8, 21),
				checkForEnemy(enemyObs, 9, 21),
				checkForEnemy(enemyObs, 10, 21),
				checkForEnemy(enemyObs, 11, 21),
				checkForEnemy(enemyObs, 12, 21),
				checkForEnemy(enemyObs, 13, 21),
				checkForEnemy(enemyObs, 14, 21),
				checkForEnemy(enemyObs, 15, 21),
				checkForEnemy(enemyObs, 16, 21),
				checkForEnemy(enemyObs, 17, 21),
				checkForEnemy(enemyObs, 18, 21),
				checkForEnemy(enemyObs, 19, 21),
				checkForEnemy(enemyObs, 20, 21),
				checkForEnemy(enemyObs, 21, 21),
				(double)yVel,
				(double)xVel,
				observation.isMarioOnGround() ? 1 : 0,
				observation.mayMarioJump() ? 1 : 0};
		
		return inputs;
	}
	
	private double checkScene(byte[][] sceneObs, int x, int y) {
		return sceneObs[x][y] != 0 ? 1 : 0;
	}
	
	private double checkForEnemy(byte[][] obs, int x, int y){
		return obs[x][y] > 0 ? 1 : 0;
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

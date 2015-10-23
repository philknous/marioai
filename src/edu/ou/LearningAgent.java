package edu.ou;

import ch.idsia.ai.agents.Agent;
import ch.idsia.mario.environments.Environment;

public interface LearningAgent extends Agent {
	
	/**
	 * Learn the desired output given the environment
	 * @param observation - the current state observation
	 * @param action - the action to be learned
	 */
	public void learn(Environment observation, boolean[] action);
	
	/**
	 * Perform learning using the reward
	 * @param reward
	 */
	public void giveReward(int reward);
}

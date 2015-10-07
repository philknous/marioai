package edu.ou;

import ch.idsia.ai.agents.ai.BasicAIAgent;
import ch.idsia.mario.environments.Environment;

public class PascualKnousAgent extends BasicAIAgent implements LearningAgent{

	public PascualKnousAgent() {
		super("PascualKnousAgent");
		
	}
	
	@Override
	public void reset() {
		
	}
	
	@Override
	public boolean[] getAction(Environment observation) {
		return action;
	}
	
	/**
	 * Learn the desired output given the environment
	 * @param observation - the current state observation
	 * @param action - the action to be learned
	 */
	public void learn(Environment observation, boolean[] action) {
		
	}
	
}

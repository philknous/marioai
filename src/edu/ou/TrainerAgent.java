package edu.ou;

import ch.idsia.ai.agents.Agent;
import ch.idsia.ai.agents.ai.BasicAIAgent;
import ch.idsia.mario.environments.Environment;

public class TrainerAgent extends BasicAIAgent {

	private Agent trainer;
	private LearningAgent trainee;
	
	public TrainerAgent(Agent trainer, LearningAgent trainee) {
		super("TrainerAgent");
		
		this.trainer = trainer;
		this.trainee = trainee;
		reset();
	}
	
	@Override
	public void reset() {
		trainer.reset();
		trainee.reset();
	}
	
	@Override
	public boolean[] getAction(Environment observation) {
		action = trainer.getAction(observation);
		
		// Teach the trainee
		trainee.learn(observation, action);
		
		return action;
	}

}

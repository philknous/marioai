package ch.idsia.ai.agents.ai;

import ch.idsia.ai.agents.Agent;
import ch.idsia.mario.environments.Environment;
import edu.ou.LearningAgent;

/**
 * Created by IntelliJ IDEA.
 * User: julian
 * Date: Aug 10, 2009
 * Time: 6:41:42 PM
 */
public class TimingAgent implements LearningAgent {

    private Agent agent;
    private long timeTaken = 0;
    private int evaluations = 0;

    public TimingAgent (Agent agent) {
        this.agent = agent;
    }

    public void reset() {
        agent.reset ();
    }

    public boolean[] getAction(Environment observation) {
        long start = System.currentTimeMillis();
        boolean[] action = agent.getAction (observation);
        timeTaken += (System.currentTimeMillis() - start);
        evaluations++;
        return action;
    }

    public AGENT_TYPE getType() {
        return agent.getType ();
    }

    public String getName() {
        return agent.getName ();
    }

    public void setName(String name) {
        agent.setName (name);
    }

    public double averageTimeTaken () {
        double average = ((double) timeTaken) / evaluations;
        timeTaken = 0;
        evaluations = 0;
        return average;
    }

	@Override
	public void learn(Environment observation, boolean[] action) {
		if (agent instanceof LearningAgent) {
			((LearningAgent)agent).learn(observation, action);
		}
	}

	@Override
	public void giveReward(int reward) {
		if (agent instanceof LearningAgent) {
			((LearningAgent)agent).giveReward(reward);
		}
		
	}
}

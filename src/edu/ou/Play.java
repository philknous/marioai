package edu.ou;

import ch.idsia.ai.agents.Agent;
import ch.idsia.ai.agents.AgentsPool;
import ch.idsia.ai.agents.ai.TimingAgent;
import ch.idsia.tools.CmdLineOptions;
import ch.idsia.tools.EvaluationInfo;
import ch.idsia.tools.EvaluationOptions;
import ch.idsia.tools.Evaluator;
import ch.idsia.utils.StatisticalSummary;

public class Play {
    final static int numberOfTrials = 10;
    private static int killsSum = 0;
    private static int marioStatusSum = 0;
    private static int timeLeftSum = 0;
    private static int marioModeSum = 0;
    private static boolean detailedStats = false;
    private static int initialSeed = 0;
    private static int initialDif = 20;

    public static void main(String[] args) {
        CmdLineOptions cmdLineOptions = new CmdLineOptions(args);

        
        cmdLineOptions.setLevelDifficulty(initialDif);
        cmdLineOptions.setVisualization(false);
        cmdLineOptions.setMaxFPS(true);
        
        int seed = initialSeed;
        
        cmdLineOptions.setLevelRandSeed(seed);
        
        EvaluationOptions evaluationOptions = cmdLineOptions;  // if none options mentioned, all defalults are used.
        createAgentsPool();

        scoreAllAgents(cmdLineOptions);

        if (cmdLineOptions.isExitProgramWhenFinished())
            System.exit(0);
    }

    private static boolean calledBefore = false;
    public static void createAgentsPool()
    {
        if (!calledBefore)
        {
        	Agent learner = new PascualKnousAgent();
            // Create an Agent here or mention the set of agents you want to be available for the framework.
            // All created agents by now are used here.
            // They can be accessed by just setting the commandline property -ag to the name of desired agent.
            calledBefore = true;
            //addAgentToThePool
            AgentsPool.addAgent(new PascualKnousAgent());
        }
    }

    public static void scoreAllAgents(CmdLineOptions cmdLineOptions)
    {
        // TODO: implement sorting with respect to CompetitionScore.
        int startingSeed = cmdLineOptions.getLevelRandSeed();
        for (Agent agent : AgentsPool.getAgentsCollection()) {
            score(agent, startingSeed, cmdLineOptions);
        }

    }


    public static void score(Agent agent, int startingSeed, CmdLineOptions cmdLineOptions) {
        TimingAgent controller = new TimingAgent (agent);
        EvaluationOptions options = cmdLineOptions;

        options.setNumberOfTrials(1);
        options.setVisualization(true);
        options.setMaxFPS(false);
        System.out.println("\nScoring controller " + agent.getName() + " with starting seed " + startingSeed);

        double competitionScore = 0;
        killsSum = 0;
        marioStatusSum = 0;
        timeLeftSum = 0;
        marioModeSum = 0;

//	        competitionScore += testConfig (controller, options, startingSeed, 0, false);
//	        competitionScore += testConfig (controller, options, startingSeed, 3, false);
//	        competitionScore += testConfig (controller, options, startingSeed, 5, false);
        competitionScore += testConfig (controller, options, startingSeed, 10, false);

        System.out.println("\nCompetition score: " + competitionScore + "\n");
        System.out.println("Number of levels cleared = " + marioStatusSum);
        System.out.println("Additional (tie-breaker) info: ");
        System.out.println("Total time left = " + timeLeftSum);
        System.out.println("Total kills = " + killsSum);
        System.out.println("Mario mode (small, large, fire) sum = " + marioModeSum);
        System.out.println("TOTAL SUM for " + agent.getName() + " = " + (competitionScore + killsSum + marioStatusSum + marioModeSum + timeLeftSum));
    }

    public static double testConfig (TimingAgent controller, EvaluationOptions options, int seed, int levelDifficulty, boolean paused) {
        options.setLevelDifficulty(levelDifficulty);
        options.setPauseWorld(paused);
        StatisticalSummary ss = test (controller, options, seed);
        double averageTimeTaken = controller.averageTimeTaken();
        System.out.printf("Difficulty %d score %.4f (avg time %.4f)\n",
                levelDifficulty, ss.mean(), averageTimeTaken);
//	        if (averageTimeTaken > 40) {
//	            System.out.println("Maximum allowed average time is 40 ms per time step.\n" +
//	                    "Controller disqualified");
//	            System.exit (0);
//	        }
        return ss.mean();
    }

    public static StatisticalSummary test (Agent controller, EvaluationOptions options, int seed) {
        StatisticalSummary ss = new StatisticalSummary ();
        int kills = 0;
        int timeLeft = 0;
        int marioMode = 0;
        int marioStatus = 0;

        options.setNumberOfTrials(numberOfTrials);
        options.resetCurrentTrial();
        for (int i = 0; i < numberOfTrials; i++) {
            options.setLevelRandSeed(seed + i);
            options.setLevelLength (200 + (i * 128) + (seed % (i + 1)));
            options.setLevelType(i % 3);
            controller.reset();
            options.setAgent(controller);
            Evaluator evaluator = new Evaluator (options);
            EvaluationInfo result = evaluator.evaluate().get(0);
            kills += result.computeKillsTotal();
            timeLeft += result.timeLeft;
            marioMode += result.marioMode;
            marioStatus += result.marioStatus;
//	            System.out.println("\ntrial # " + i);
//	            System.out.println("result.timeLeft = " + result.timeLeft);
//	            System.out.println("result.marioMode = " + result.marioMode);
//	            System.out.println("result.marioStatus = " + result.marioStatus);
//	            System.out.println("result.computeKillsTotal() = " + result.computeKillsTotal());
            ss.add (result.computeDistancePassed());
        }

        if (detailedStats)
        {
            System.out.println("\n===================\nStatistics over " + numberOfTrials + " trials for " + controller.getName());
            System.out.println("Total kills = " + kills);
            System.out.println("marioStatus = " + marioStatus);
            System.out.println("timeLeft = " + timeLeft);
            System.out.println("marioMode = " + marioMode);
            System.out.println("===================\n");
        }

        killsSum += kills;
        marioStatusSum += marioStatus;
        timeLeftSum += timeLeft;
        marioModeSum += marioMode;

        return ss;
    }
}

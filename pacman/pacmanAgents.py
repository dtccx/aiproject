# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from heuristics import *
import random
import collections as cls

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        global scored
        scored = []
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        q = cls.deque()
        # timeOfNextSteps
        time = 0
        q.append((state, Directions.STOP, time))
        isBreak = False
        while q:
            cur, curAction, time = q.popleft()
            # time += 1
            if (cur.isWin()):
                return curAction
            if (cur.isLose()):
                continue
            legal = cur.getLegalPacmanActions()
            # return action, state
            successors = [(cur.generatePacmanSuccessor(action), action) for action in legal]
            # global scored
            # scored = scored + [(admissibleHeuristic(state), action) for state, action in successors]
            # scored.add(scoredTemp)
            for state, action in successors:
                if (state != None):
                    if (state.isWin()):
                        return curAction
                    if (state.isLose()):
                        continue
                    q.append((state, action, time + 1))
                else:
                    isBreak = True
                    break
            if isBreak:
                break
        # global scored
        # print(q)

        #If not reaching a terminal state, return the action leading to the node with
        #the min score and no children based on the heuristic function
        scored = [(admissibleHeuristic(state), action, time) for state, action, time in q]
        # print(scored,"/n")
        if scored == []:
            return Directions.STOP
        bestScore = min(scored)[0]
        for score, action, time in scored:
            if score == bestScore:
                bestAction = action
                print(bestAction)
                break
        bestArray = min(scored[0], scored[2])
        # bestArray = max(scored, key=lambda s: (s[0], -s[2]))
        # bestAction = bestArray[1]
        #print(bestArray)
        #print(bestAction)
        return bestAction

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        return Directions.STOP

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        return Directions.STOP

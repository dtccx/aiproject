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
import math

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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        # nothing Initialization
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        possible = state.getAllPossibleActions()
        flag = False
        # create action sequence
        actSeq = list()
        finScore = [gameEvaluation(state, state), [Directions.STOP]]
        for i in range(5):
            actSeq.append(random.choice(possible))

        while True:
            actionNext = list()
            for i in range(5):
                if (random.randint(0, 1) == 0):
                    actionNext.append(actSeq[i])
                else:
                    actionNext.append(random.choice(possible))
            stateCur = state
            # perform all the sequence
            for action in actionNext:
                stateNext = stateCur.generatePacmanSuccessor(action)
                if stateNext is not None:
                    if not stateNext.isWin() and not stateNext.isLose():
                        stateCur = stateNext
                    # if win, just return.. (if comment this, it also works)
                    if stateNext.isWin():
                        return actionNext[0]
                    if stateNext.isLose():
                        break
                    continue
                else:
                    flag = True
                    break
            if flag == True:
                return finScore[1][0]
            score = gameEvaluation(state, stateCur)
            if (score > finScore[0]):
                finScore[0] = score
                finScore[1] = actionNext
                # finScore[1].append(actionNext[0])
                actSeq = actionNext
            else:
                continue

        # return Directions.STOP

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        # DataStructure:
        #  0                1       2
        # [actionsequence, score, rank]

        population = list()
        possible = state.getAllPossibleActions()
        for i in range(8):





        return Directions.STOP

class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        #       0       1       2       3       4
        # NODE (action, child, score, parent, visited)
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # Data Structure (instead)
            # index  0       1       2       3       4
            # NODE (action, child, score, parent, visited)
        # TODO: write MCTS Algorithm instead of returning Directions.STOP

        # count the UCT score to get best score child
        def UCT(node, visitTimes):
            # socre / visited
            return node[2] / float(node[4]) + (math.sqrt(2 * math.log(float(visitTimes)) / float(node[4])))

        def fullExpand(node, state):
            actions = state.getLegalPacmanActions()
            if (len(actions) == len(node[1])):
                return True
            return False

        def treePolicy(node, state):
            # Check if the current node is the leaf node
            while True:
                if fullExpand(node, state):
                    # get best score child
                    node = max(node[1], key = lambda x: UCT(x, node[4]))
                else:
                    return expand(node, state)
            return node

        def expand(node, state):
            childAction = [child[0] for child in node[1]]
            actions = state.getLegalPacmanActions()
            for action in actions:
                stateNext = state.generatePacmanSuccessor(action)
                if stateNext is None:
                    return None
                if stateNext.isLose():
                    return None
                if action not in childAction:
                    newNode = [action, [], 0.0, node, 1]
                    node[1].append(newNode)
                    return newNode
            return None

        def defaultPolicy(node, state):
            # roll out 5 times
            curState = state
            actions = state.getLegalPacmanActions()
            if len(actions) == 0:
                return None
            for i in range(5):
                randomAction = random.choice(actions)
                curState = curState.generatePacmanSuccessor(randomAction)
                if curState == None:
                    return None
                if curState.isLose():
                    return None
            reward = gameEvaluation(rootState, curState)
            return reward

        def bp(node, reward):
            while node != None:
                node[4] = node[4] + 1
                node[2] = node[2] + reward
                node = node[3]
            return

        # Main Funtion:
        rootState = state
        root = [None, [], 0.0, None, 1]
        while True:
            expandNode = treePolicy(root, state)
            if expandNode == None:
                break
            reward = defaultPolicy(expandNode, state)
            if reward == None:
                break
            bp(expandNode, reward)

        if len(root[1]) == 0:
            return Directions.STOP
        # get most visited node
        bestVisit = max(root[1], key=lambda x: x[4])[4]
        bestAction = [child[0] for child in root[1] if child[4] == bestVisit]

        return random.choice(bestAction)

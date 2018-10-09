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
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        # queue store (state, action) stores the next state after the exact one after current state action and the action
        # (the action will stay same)
        q = cls.deque()
        for action in state.getLegalPacmanActions():
            q.append((state.generatePacmanSuccessor(action), action))
        isBreak = False
        while q:
            curState, preAction = q.popleft()
            if (curState.isWin()):
                return preAction
            if (curState.isLose()):
                continue
            # curState.generatePacmanSuccessor(curAction) means the nextState
            successors = [(curState.generatePacmanSuccessor(curAction), curAction) for curAction in curState.getLegalPacmanActions()]
            # scored = scored + [(admissibleHeuristic(state), action) for state, action in successors]
            # scored.add(scoredTemp)
            for nextState, curAction in successors:
                if (nextState != None):
                    # if (nextState.isWin()):
                    #     return curAction
                    # if (nextState.isLose()):
                    #     continue
                    q.append((nextState, preAction))
                else:
                    isBreak = True
                    break
            if isBreak:
                break

        #If you did not reach a terminal state, return the action leading to the node with the minimum total cost.
        scored = [(admissibleHeuristic(state), action) for state, action in q]
        # print(scored,"/n")
        if scored == []:
            return Directions.STOP
        bestScore = min(scored)[0]
        for score, action in scored:
            if (score == bestScore):
                bestAction = action
                print(bestAction)
                break
        return bestAction
        # bestArray = min(scored, key=lambda s: (s[0], s[2]))
        # bestArray = max(scored, key=lambda s: (s[0], -s[2]))
        # bestAction = bestArray[1]
        #print(bestArray)

        # bestScore = min(scored)[0]
        # # get all actions that lead to the highest score
        # bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # # return random action from the list of the best actions
        # return random.choice(bestActions)


class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):

        return

    def dfs(self, state, action, list, tempList):
        if (state == None):
            if tempList:
                tempList.pop()
            list.extend(tempList)
            return

        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        for nextState, curAction in successors:
            # if (nextState != None):
            # if (nextState == None):
            #     return
            tempList.append((nextState, action))
            self.dfs(nextState, curAction, list, tempList)
            if tempList:
                tempList.pop()
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        list = []
        # Legal action
        for action in state.getLegalPacmanActions():
            self.dfs(state.generatePacmanSuccessor(action), action, list, [])
        #self.dfs(state, None, list)
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in list]
        if scored == []:
            return Directions.STOP
        bestScore = min(scored)[0]
        for score, action in scored:
            if (score == bestScore):
                bestAction = action
                print(bestAction)
                break
        return bestAction


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        return Directions.STOP

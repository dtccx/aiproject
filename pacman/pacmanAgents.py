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
import heapq as hq

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
        depth = 0
        for action in state.getLegalPacmanActions():
            q.append((state.generatePacmanSuccessor(action), action, depth))
        isBreak = False
        while q:
            curState, preAction, depth = q.popleft()
            if (curState.isWin()):
                return preAction
            if (curState.isLose()):
                continue
            depth += 1
            # curState.generatePacmanSuccessor(curAction) means the nextState
            successors = [(curState.generatePacmanSuccessor(curAction), curAction) for curAction in curState.getLegalPacmanActions()]
            # scored = scored + [(admissibleHeuristic(state), action) for state, action in successors]
            # scored.add(scoredTemp)
            for nextState, curAction in successors:
                if (nextState != None):
                    q.append((nextState, preAction, depth))
                else:
                    isBreak = True
                    break
            if isBreak:
                break

        #If you did not reach a terminal state, return the action leading to the node with the minimum total cost.
        scored = [(admissibleHeuristic(state) + depth, action) for state, action, depth in q]
        # print(scored,"/n")
        if scored == []:
            return Directions.STOP
        bestScore = min(scored, key=lambda s: s[0])[0]
        for score, action in scored:
            if (score == bestScore):
                bestAction = action
                # print(bestAction)
                break
        return bestAction



class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        # global visited
        # visited = set()
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        stack = cls.deque()
        visited = set()
        visited.add(state)
        depth = 0
        for action in state.getLegalPacmanActions():
            newState = state.generatePacmanSuccessor(action)
            stack.append((newState, action, depth))
            # visited.add(newState)
        isBreak = False
        while stack:
            curState, preAction, depth = stack.pop()
            if (curState.isWin()):
                return preAction
            if (curState.isLose()):
                continue
            if curState in visited:
                continue
            visited.add(curState)
            depth += 1
            # curState.generatePacmanSuccessor(curAction) means the nextState
            successors = [(curState.generatePacmanSuccessor(curAction), curAction) for curAction in curState.getLegalPacmanActions()]
            for nextState, curAction in successors:
                if (nextState != None):
                    stack.append((nextState, preAction, depth))
                else:
                    isBreak = True
                    break
            if isBreak:
                break

        scored = [(admissibleHeuristic(state) + depth, action) for state, action, depth in stack]
        if scored == []:
            return Directions.STOP
        # bestAction = min(scored, key=lambda s: s[0])[1]
        # return bestAction

        bestScore = min(scored)[0]
        # print(bestScore)
        # for score, action in scored:
        #     if (score == bestScore):
        #         bestAction = action
        #         print(bestAction)
        #         break
        # return bestAction
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]

        # return random action from the list of the best actions
        return random.choice(bestActions)
        #return bestAction


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        visited = set()
        heap = []
        depth = 0
        for action in state.getLegalPacmanActions():
            hq.heappush(heap, (admissibleHeuristic(state.generatePacmanSuccessor(action)) + depth, state.generatePacmanSuccessor(action), action, depth))
        isBreak = False
        while heap:
            score, curState, preAction, depth = hq.heappop(heap)
            if (curState.isWin()):
                print("win")
                return preAction
            if (curState.isLose()):
                continue
            if curState in visited:
                continue
            visited.add(curState)

            # print("score", score)
            depth += 1
            # curState.generatePacmanSuccessor(curAction) means the nextState
            successors = [(curState.generatePacmanSuccessor(curAction), curAction) for curAction in curState.getLegalPacmanActions()]
            for nextState, curAction in successors:
                if (nextState != None):
                    hq.heappush(heap, (admissibleHeuristic(nextState) + depth, nextState, preAction, depth))
                else:
                    isBreak = True
                    break
            if isBreak:
                break

        if heap == []:
            return Directions.STOP
        score, curState, preAction, depth = hq.heappop(heap)
        return preAction

        # scored = [(admissibleHeuristic(state) + depth, action) for state, action, depth in list]
        # if scored == []:
        #     return Directions.STOP
        # bestScore = min(scored)[0]
        # for score, action in scored:
        #     if (score == bestScore):
        #         bestAction = action
        #         print(bestAction)
        #         break
        # return bestAction
        # bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # # return random action from the list of the best actions
        # return random.choice(bestActions)
        #
        # return Directions.STOP

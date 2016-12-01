# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        returnScoreElements = []

        # # starting game score
        # gameStateScore = successorGameState.getScore()
        # returnScoreElements.append(gameStateScore)

        # number of food pellets remaining - less is better
        numFood = 0
        for food in newFood.asList():
            if food:
                numFood += 1
        returnScoreElements.append(- 10000 * numFood)

        if numFood == 0:
            return 1000000

        # max dist to ghost - more is better
        maxDistanceToGhost = 0
        for state in newGhostStates:
            distance = manhattanDistance(ghostState.configuration.pos, newPos)
            if maxDistanceToGhost < distance:
                maxDistanceToGhost = distance

        returnScoreElements.append(maxDistanceToGhost / 2)

        # if pacman is right next to a ghost, flee
        if maxDistanceToGhost <= 2:
            return -10000000

        # # max dist to food - less is better
        # maxDistanceToFood = 0
        # for food in newFood.asList():
        #     distance = manhattanDistance(food, newPos) + 2
        #     if maxDistanceToFood < distance:
        #         maxDistanceToFood = distance
        #
        # returnScoreElements.append(-maxDistanceToFood)

        # min dist to food - less is better
        minDistanceToFood = 1000000
        for food in newFood.asList():
            distance = manhattanDistance(food, newPos) + 2
            if minDistanceToFood > distance:
                minDistanceToFood = distance

        returnScoreElements.append(- 100 * minDistanceToFood)

        returnScore = 0
        for element in returnScoreElements:
            returnScore += element

        return returnScore


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxValue(self, gameState, agentIndex, myDepth):
        v = -1e309
        legalActions = gameState.getLegalActions(agentIndex)

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action);
            v = max(v, self.stateValue(successorState, myDepth + 1))

        return v

    def minValue(self, gameState, agentIndex, myDepth):
        v = 1e309

        legalActions = gameState.getLegalActions(agentIndex)

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action);
            v = min(v, self.stateValue(successorState, myDepth + 1))

        return v

    def stateValue(self, gameState, myDepth):
        # BASE CASE
        calculatedDepth = (myDepth / gameState.getNumAgents())

        if gameState.isLose() or gameState.isWin() or calculatedDepth >= self.depth:
            return self.evaluationFunction(gameState)

        # RECURSIVE CASES

        # the agent that's about to move
        agentIndex = myDepth % gameState.getNumAgents()

        # if pacman is about to move
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, myDepth)

        # if a ghost is about to move
        else:
            return self.minValue(gameState, agentIndex, myDepth)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        bestStateValue = -1e309
        bestAction = None

        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            successorStateValue = self.stateValue(successorState, 1)
            if successorStateValue > bestStateValue:
                bestAction = action
                bestStateValue = successorStateValue

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, agentIndex, myDepth, alpha, beta):
        v = -1e309
        legalActions = gameState.getLegalActions(agentIndex)

        bestAction = None

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action);
            successorStateValue = self.stateValue(successorState, myDepth + 1, alpha, beta)

            if successorStateValue[1] > v:
                v = successorStateValue[1]
                bestAction = action

            if v > beta:
                return [bestAction, v]
            alpha = max(alpha, v)

        return [bestAction, v]

    def minValue(self, gameState, agentIndex, myDepth, alpha, beta):
        v = 1e309
        legalActions = gameState.getLegalActions(agentIndex)

        bestAction = None

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action);
            successorStateValue = self.stateValue(successorState, myDepth + 1, alpha, beta)

            if successorStateValue[1] < v:
                v = successorStateValue[1]
                bestAction = action

            if v < alpha:
                return [bestAction, v]
            beta = min(beta, v)

        return [bestAction, v]

    def stateValue(self, gameState, myDepth, alpha, beta):
        # BASE CASE
        calculatedDepth = (myDepth / gameState.getNumAgents())

        if gameState.isLose() or gameState.isWin() or calculatedDepth >= self.depth:
            return [None, self.evaluationFunction(gameState)]

        # RECURSIVE CASES

        # the agent that's about to move
        agentIndex = myDepth % gameState.getNumAgents()

        # if pacman is about to move
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, myDepth, alpha, beta)

        # if a ghost is about to move
        else:
            return self.minValue(gameState, agentIndex, myDepth, alpha, beta)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        alpha = -1e309
        beta = 1e309

        [bestAction, value] = self.stateValue(gameState, 0, alpha, beta)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValueRand(self, gameState, agentIndex, myDepth):
        v = -1e309
        legalActions = gameState.getLegalActions(agentIndex)

        bestActions = []

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action);
            successorStateValue = self.stateValue(successorState, myDepth + 1)[1]

            if successorStateValue == v:
                bestActions.append(action)

            if successorStateValue > v:
                v = successorStateValue
                bestActions = [action]

        from random import randint
        index = randint(0, len(bestActions) - 1)

        bestAction = bestActions[index]

        return [bestAction, v]

    def maxValue(self, gameState, agentIndex, myDepth):
        v = -1e309
        legalActions = gameState.getLegalActions(agentIndex)

        bestAction = None

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action);
            successorStateValue = self.stateValue(successorState, myDepth + 1)[1]

            if successorStateValue > v:
                v = successorStateValue
                bestAction = action

        return [bestAction, v]

    def avgValue(self, gameState, agentIndex, myDepth):
        legalActions = gameState.getLegalActions(agentIndex)

        sum = 0.0

        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action);
            successorStateValue = self.stateValue(successorState, myDepth + 1)[1]

            sum += float(successorStateValue)

        avg = sum / len(legalActions)

        return [None, avg]

    def stateValue(self, gameState, myDepth):
        # BASE CASE
        calculatedDepth = (myDepth / gameState.getNumAgents())

        if gameState.isLose() or gameState.isWin() or calculatedDepth >= self.depth:
            return [None, self.evaluationFunction(gameState)]

        # RECURSIVE CASES

        # the agent that's about to move
        agentIndex = myDepth % gameState.getNumAgents()

        # if pacman is about to move
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, myDepth)

        # if a ghost is about to move
        else:
            return self.avgValue(gameState, agentIndex, myDepth)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        [bestAction, value] = self.stateValue(gameState, 0)

        return bestAction

def betterEvaluationFunction(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    returnScoreElements = []

    # # starting game score
    # gameStateScore = successorGameState.getScore()
    # returnScoreElements.append(gameStateScore)

    # number of food pellets remaining - less is better
    numFood = 0
    for food in newFood.asList():
        if food:
            numFood += 1
    returnScoreElements.append(- 10000 * numFood)

    if numFood == 0:
        return 1000000

    # max dist to ghost - more is better
    maxDistanceToGhost = 0
    for state in newGhostStates:
        distance = manhattanDistance(state.configuration.pos, newPos)
        if maxDistanceToGhost < distance:
            maxDistanceToGhost = distance

    returnScoreElements.append(maxDistanceToGhost / 2)

    # if pacman is right next to a ghost, flee
    if maxDistanceToGhost <= 2:
        return -10000000

    # # max dist to food - less is better
    # maxDistanceToFood = 0
    # for food in newFood.asList():
    #     distance = manhattanDistance(food, newPos) + 2
    #     if maxDistanceToFood < distance:
    #         maxDistanceToFood = distance
    #
    # returnScoreElements.append(-maxDistanceToFood)

    # min dist to food - less is better
    minDistanceToFood = 1000000
    for food in newFood.asList():
        distance = manhattanDistance(food, newPos) + 2
        if minDistanceToFood > distance:
            minDistanceToFood = distance

    returnScoreElements.append(- 100 * minDistanceToFood)

    returnScore = 0
    for element in returnScoreElements:
        returnScore += element

    return returnScore
#
# def betterEvaluationFunctionTry(currentGameState):
#     """
#       Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#       evaluation function (question 5).
#
#       DESCRIPTION: <write something here so we know what you did>
#     """
#
#     pacmanPosition = currentGameState.getPacmanPosition()
#     foods = currentGameState.getFood()
#     ghostStates = currentGameState.getGhostStates()
#     capsules = currentGameState.getCapsules()
#     scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
#     gameStateScore = currentGameState.getScore()
#     numFood = currentGameState.getNumFood()
#     ghostStates = currentGameState.getGhostStates()
#
#     returnScoreElements = []
#
#     # max dist to ghost - more is better
#     maxDistanceToGhost = 0
#     for ghostState in ghostStates:
#         if ghostState.scaredTimer > 0:
#             maxDistanceToGhost = 1e309
#         else:
#             distance = manhattanDistance(ghostState.getPosition(), pacmanPosition)
#             if maxDistanceToGhost < distance:
#                 maxDistanceToGhost = distance
#
#     # max dist to food - less is better
#     maxDistanceToFood = 0
#     for food in foods.asList():
#         distance = manhattanDistance(food, pacmanPosition) + 2
#         if maxDistanceToFood < distance:
#             maxDistanceToFood = distance
#
#     # sum of all distances to food
#     sumDistanceToFood = 0
#     for food in foods.asList():
#         sumDistanceToFood += manhattanDistance(food, pacmanPosition) + 2
#
#     # min dist to food - less is better
#     minDistanceToFood = 1e309
#     for food in foods.asList():
#         distance = manhattanDistance(food, pacmanPosition) + 2
#         if minDistanceToFood > distance:
#             minDistanceToFood = distance
#
#     minDistanceToCapsule = 1e309
#     for capsule in capsules:
#         distance = manhattanDistance(capsule, pacmanPosition) + 2
#         if minDistanceToCapsule > distance:
#             minDistanceToCapsule = distance
#
#     closestFoodMazeDistance = mazeDistance(currentGameState)
#
#     returnScoreElements.append(maxDistanceToGhost)
#     returnScoreElements.append(-100 * numFood)
#     # returnScoreElements.append(-20 * closestFoodMazeDistance)
#     # returnScoreElements.append(-10 * minDistanceToFood)
#     # returnScoreElements.append(-1 * maxDistanceToFood)
#     # returnScoreElements.append(-1 * minDistanceToCapsule)
#     returnScoreElements.append(-10 * len(capsules))
#     returnScoreElements.append(-5 * gameStateScore)
#
#     if numFood == 0:
#         return 1e309
#
#     # if pacman is right next to a ghost, flee
#     if maxDistanceToGhost <= 2:
#         return -1e309
#
#     returnScore = 0
#     for element in returnScoreElements:
#         returnScore += element
#
#     return returnScore
#
#
# # Abbreviation
# better = betterEvaluationFunction
#
# from game import Actions
#
#
# class FoodSearchProblem:
#     """
#     A search problem associated with finding the a path that collects all of the
#     food (dots) in a Pacman game.
#
#     A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
#       pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
#       foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
#     """
#
#     def __init__(self, startingGameState):
#         self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
#         self.walls = startingGameState.getWalls()
#         self.startingGameState = startingGameState
#         self._expanded = 0  # DO NOT CHANGE
#         self.heuristicInfo = {}  # A dictionary for the heuristic to store information
#
#     def getStartState(self):
#         return self.start
#
#     def isGoalState(self, state):
#
#         oldFoodList = self.start[1].asList()
#         numOldFood = len(oldFoodList)
#
#         newFoodList = state[1].asList()
#         numNewFood = len(newFoodList)
#
#         return numNewFood < numOldFood
#
#     def getSuccessors(self, state):
#         "Returns successor states, the actions they require, and a cost of 1."
#         successors = []
#         self._expanded += 1  # DO NOT CHANGE
#         for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
#             x, y = state[0]
#             dx, dy = Actions.directionToVector(direction)
#             nextx, nexty = int(x + dx), int(y + dy)
#             if not self.walls[nextx][nexty]:
#                 nextFood = state[1].copy()
#                 nextFood[nextx][nexty] = False
#                 successors.append((((nextx, nexty), nextFood), direction, 1))
#         return successors
#
#     def getCostOfActions(self, actions):
#         """Returns the cost of a particular sequence of actions.  If those actions
#         include an illegal move, return 999999"""
#         x, y = self.getStartState()[0]
#         cost = 0
#         for action in actions:
#             # figure out the next state and see whether it's legal
#             dx, dy = Actions.directionToVector(action)
#             x, y = int(x + dx), int(y + dy)
#             if self.walls[x][y]:
#                 return 999999
#             cost += 1
#         return cost
#
#
# def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
#     "*** YOUR CODE HERE ***"
#
#     # closed <- an empty set
#     closed = set()
#
#     # fringe <- insert(make-node(initial-state[problem]), fringe)
#     fringe = util.Queue()
#     fringe.push([problem.getStartState(), []])
#
#     while True:
#         # if fringe is empty then return failure
#         if fringe.isEmpty():
#             return None
#
#         # node <- remove-front(fringe, strategy)
#         node, actions = fringe.pop()
#
#         # if goal-test(problem, state[node])
#         if problem.isGoalState(node):
#             return actions
#
#         # if state[node] is not in closed
#         if node not in closed:
#             # add state[node] to closed
#             closed.add(node)
#
#             # for child-node in expand(state[node], problem)
#             for successor, action, stepCost in problem.getSuccessors(node):
#                 # fringe <- insert(child-node, fringe)
#
#                 fringe.push((successor, actions + [action]))
#
#
# def mazeDistance(gameState):
#     """
#     Returns the maze distance between pacman and the closest food pellet
#     you have already built. The gameState can be any game state -- Pacman's
#     position in that state is ignored.
#
#     Example usage: mazeDistance( (2,4), (5,6), gameState)
#
#     This might be a useful helper function for your ApproximateSearchAgent.
#     """
#     # x1, y1 = point1
#     # x2, y2 = point2
#     # walls = gameState.getWalls()
#     # assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
#     # assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
#     # prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
#
#     prob = FoodSearchProblem(gameState)
#     path = breadthFirstSearch(prob)
#     if path is None:
#         return 0
#
#     return len(path)



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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # print(newGhostStates)
        distsToFood = [manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFood = min(distsToFood) if distsToFood else 1

        foodScore = 1 / closestFood

        ghostScore = 0
        for ghost in newGhostStates:
            scaredTime = ghost.scaredTimer
            distToGhost = manhattanDistance(newPos, ghost.getPosition())

            if scaredTime == 0 and distToGhost < 2:
                ghostScore -= 10
            elif scaredTime > 0:
                ghostScore += 2 / (distToGhost + 1)

        totalScore = foodScore + ghostScore

        return successorGameState.getScore() + totalScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimaxVal(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            values = [minimaxVal(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth) for action in legalActions]
            return max(values) if agentIndex == 0 else min (values)
        
        bestScore = float('-inf')
        bestAction = None
        
        for action in gameState.getLegalActions(0):
            score = minimaxVal(gameState.generateSuccessor(0, action), 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        
        return bestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def abPruning(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, abPruning(successor, nextAgent, nextDepth, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                value = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, abPruning(successor, nextAgent, nextDepth, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        bestScore = float('-inf')
        bestAction = None
        alpha, beta = float('-inf'), float('inf')

        for action in gameState.getLegalActions(0):
            score = abPruning(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimaxVal(state, agentIndex, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            actions = state.getLegalActions(agentIndex)

            if agentIndex == 0:
                value = float('-inf')
                for move in actions:
                    successor = state.generateSuccessor(agentIndex, move)
                    value = max(value, expectimaxVal(successor, nextAgent, nextDepth))
                return value
            else:
                value = 0
                prob = 1.0 / len(actions)
                for move in actions:
                    successor = state.generateSuccessor(agentIndex, move)
                    value += prob * expectimaxVal(successor, nextAgent, nextDepth)
                return value

        bestScore = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimaxVal(successor, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    ghostPos = [ghost.getPosition() for ghost in ghosts]
    scaredGhost = [ghost.scaredTimer > 0 for ghost in ghosts]
    score = currentGameState.getScore()

    leftFoods = len(foods)
    foodDist = min(manhattanDistance(currentPos, food)for food in foods) if foods else 0.5

    ghostDist = float('inf')
    for i, ghostPosition in enumerate(ghostPos):
        distance = manhattanDistance(currentPos, ghostPosition)
        if scaredGhost[i]:
            ghostDist = min(ghostDist, distance)
        else:
            if distance < 2:
                ghostDist = min(ghostDist, distance)

    finalScore = score + (1.0 / (foodDist + 1)) * 10

    if ghostDist < 2:
        finalScore -= 1000
    else:
        finalScore += 1.0 / (ghostDist + 1)
    
    if leftFoods == 0:
        finalScore += 500
    if leftFoods > 5:
        finalScore -= 5 * (leftFoods - 5)
    if ghostDist < 3 and any(scaredGhost):
        finalScore += 200

    return finalScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

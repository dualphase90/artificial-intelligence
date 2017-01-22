"""
This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using run_tournament.py and include the results in your
report.
"""
from random import randint
import random
random.seed(4)

class Timeout(Exception):
    """ Subclass base exception for code clarity. """
    pass


class CustomEval():
    """
    Custom evaluation function that acts however you think it should.
    """

    def score(self, game, player):
        """
        Calculate the heuristic value of a game state from the point of view of
        the given player.

        Args:
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : hashable
            One of the objects registered by the game object as a valid player.
            (i.e., `player` should be either game.__player_1__ or
            game.__player_2__).

        Returns:
        ----------
        float
            The heuristic value of the current game state.
        """
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        x=1
        y=1 
        score=float(x*(own_moves)-(y*opp_moves))
        #score= float(own_moves+1)/(opp_moves+1)
        player_num= 1 if game.__player_1__==player else 2
        #x=[0, 0.2, 0.4, 0.6, 0.8, 1]
        #y=[0, 0.2, 0.4, 0.6, 0.8, 1]        
        #print("player location: {:s}  Score : {:s}  player number : {:d}  ".format(str(game.get_player_location(player)),str(score),player_num))
        #print("\tLegal moves"+str(game.get_legal_moves(player)))

        # TODO: finish this function!
        if game.is_winner(player):
        	return float("inf")

        elif game.is_loser(player):
        	return float("-inf")
        
        else:
            
             
             return float(score)#*float(len(game.get_blank_spaces())))
           


        #raise NotImplementedError

class CustomPlayer():
    """
    Game-playing agent that chooses a move using your evaluation function and a
    depth-limited minimax algorithm with alpha-beta pruning. You must finish
    and test this player to make sure it properly uses minimax and alpha-beta
    to return a good move before the search time limit expires.

    Args:
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. This is
        defined such that a depth of one (1) would only explore the immediate
        sucessors of the current state.

    eval_fn : class (optional)
        List of the legal moves available to the player with initiative to
        move in the current game state (this player).

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().
    """

   

    def __init__(self, search_depth=2,eval_fn=CustomEval(), iterative=False, method='alphabeta'):
        """
        You MAY modify this function, but the interface must remain compatible
        with the version provided.
        """
        self.node_count=0
        #self.node_count_AB=0
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.iterative = iterative
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = 10  # time (in ms) to leave on the clock when terminating search
        #self.search_depth=3
        print("search depth",self.search_depth)



    def get_move(self, game, legal_moves, time_left):
        """d
        Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return before the
              timer reaches 0.
        **********************************************************************

        Args:
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns:
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!

        
        # Perform any required initializations
        if not legal_moves:
            return (-1, -1)


        try:
            # print(legal_moves)

            #print("------Before Minimax-------")
            #print(game.print_board())
            #self.method="alphabeta" if self.method== "alphabeta" else "minimax"   
            if self.method=="alphabeta":
                score,move=(self.alphabeta(game,depth=self.search_depth,maximizing_player=True))	
            else:
                score,move=(self.minimax(game,depth=self.search_depth,maximizing_player=True))    
           
            #print("legal moves      ",legal_moves)
           # print("scores   ",score,move)
            #print("score for the current move",self.eval_fn.score(game, game.active_player))
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
			   


           

        except Timeout:

            # Handle any actions required at timeout, if necessary
       	    print('-- TIMEOUT --')


        # Return the best move from the last completed search iteration
        #print(scores)
        #print("scores   =",best_move)
        return move

        #raise NotImplementedError
        #return legal_moves[0] if legal_moves else (-1, -1)

        
    def minimax(self, game, depth, maximizing_player=True):
        """
        Implement the minimax search algorithm as described in the lectures.

        **********************************************************************
        NOTE: You may modify the function signature and/or output, but the
              signature must remain compatible with the version provided.
              (i.e., if you add parameters, you must also set defaults.) The
              project reviewers will evaluate your code with a test suite
              that depends on the provided input interface. (The output
              signature can be changed, as it is not used for testing.)
        **********************************************************************

        Args:
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns:
        ----------
        float
            By default, the output at least returns the floating point value
            for the min or max score associated with the moves in this branch
            of the game tree.

            YOU ARE ALLOWED TO CHANGE THE OUTPUT INTERFACE OF THIS FUNCTION
        """
        

       # if self.time_left() < self.TIMER_THRESHOLD:
        #    raise Timeout()
        
        scores = []
        

        legal_moves = game.get_legal_moves()
        player=game.active_player if maximizing_player else game.inactive_player
             
        # BASE CASE
        if not legal_moves or depth == 0:	
           # print("Base case",self.eval_fn.score(game, game.active_player))		        		
            return (self.eval_fn.score(game, player),(-1,-1))


        # RECURSIVE CASE
        for move in legal_moves:
            board = game.forecast_move(move)
            #print("Blank spaces",len(game.get_blank_spaces()))
            score,bestMove=(self.minimax(board, depth=depth-1, maximizing_player =  not maximizing_player))
            scores.append((score, move))
                	
        if maximizing_player:
            best_move = sorted(scores,reverse=True)[0]        
            return best_move
        else:
             best_move = sorted(scores,reverse=False)[0]        
             return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        # if maximizing, player is active player; else, opponent
        player = game.active_player

        # for either case, we're going to need to get their list of legal moves
        moves = game.get_legal_moves()

        # base case: go either to depth or the end of the game (no child nodes)
        if depth == 0 or not moves:
            evalFrom = player if maximizing_player else game.get_opponent(player)
            score = self.eval_fn.score(game, evalFrom)
            return (score,(-1,-1))
        
        bestMove=(-1,3)
        # max or min? max: max values, next is min; vice versa
        if maximizing_player:
            bestVal = float('-inf')
            for move in moves:
                childBoard = game.forecast_move(move)
                moveVal,currentMove = self.alphabeta(childBoard, depth - 1, alpha=alpha, beta=beta, maximizing_player=False)
                #print(moveVal,bestMove)
                if moveVal>bestVal:
                    bestVal=moveVal
                    bestMove=move

                
                #print(moveVal,bestMove)
               
                alpha = max(alpha, bestVal) # new: check against current alpha


                # if bestVal>bestVal:
                #     bestVal=moveVal
                #     bestMove=currentMove
                
                #print(bestVal,beta)
                if bestVal>=beta: break # new: prune if there's an opportunity


            return bestVal,bestMove
        else: # minimizing_player
            bestVal = float('inf')
            for move in moves:
                childBoard = game.forecast_move(move)
                moveVal,currentMove = self.alphabeta(childBoard, depth - 1, alpha=alpha, beta=beta)
                #bestVal = min(bestVal, moveVal)
                if moveVal<bestVal:
                    bestVal=moveVal
                    bestMove=move
                beta = min(beta, bestVal) # new: check against current beta
                if bestVal <= alpha: break # new: prune if there's an opportunity
            return bestVal,bestMove
        
        """
        Implement minimax search with alpha-beta pruning as described in the
        lectures.

        **********************************************************************
        NOTE: You may modify the function signature and/or output, but the
              signature must remain compatible with the version provided.
              (i.e., if you add parameters, you must also set defaults.) The
              project reviewers will evaluate your code with a test suite
              that depends on the provided input interface. (The output
              signature can be changed, as it is not used for testing.)
        **********************************************************************
        Args:
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns:
        ----------
        float
            By default, the output at least returns the floating point value
            for the min or max score associated with the moves in this branch
            of the game tree.

            YOU ARE ALLOWED TO CHANGE THE OUTPUT INTERFACE OF THIS FUNCTION
        """
    """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

    # TODO: finish this function!      
    
    
        # BASE CASE
        if not legal_moves or depth == 0:   
           # print("Base case",self.eval_fn.score(game, game.active_player))                        
            return self.eval_fn.score(game, player)


        # RECURSUVE CASE for the maximizer
        def max_value(game, alpha, beta, depth):
              
           #v= best value seen so far at previous node
            v = -infinity
            moves = game.get_legal_moves(player)
            for move in moves:
                game.forecast_move(move)
                score.append
         #determinine maximum value in next state
            v = max(v, min_value(game.for = in xrange(1,10):
                passself.eval_fn.score(game, player), alpha, beta, depth-1))
            #Prune if current value better than next value
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
       

        def min_value(game, alpha, beta, depth):  
            #v= best value seen so far at previous node
            v = infinity
            moves = game.get_legal_moves(player)
            for move in moves:
                game.forecast_move(move)
    
            #determinine maximum value in next state
            v = min(v, max_value(eval_fn(game), alpha, beta, depth-1))
            #Prune if current value better than next value
            move = self.alphabeta(game.forecast_move(move), depth - 1, alpha=alpha, beta=beta, maximizing_player=False)               
            if v <= alpha:
                return v
            beta = min(beta, v)
            return v        
    

        if maximizing_player:
            max_value(game,alpha,beta,depth)
        else:
            min_value(game,alpha,beta,depth)
    

         return move
    """
    """
    """
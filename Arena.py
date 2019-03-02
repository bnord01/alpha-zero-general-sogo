import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.game_lengths = []

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.init_board()
        it = 0
        while self.game.terminal_value(board, current_player)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(current_player))
                self.display(board)
            action = players[current_player+1](self.game.canonical_board(board, current_player))

            valids = self.game.valid_actions(self.game.canonical_board(board, current_player),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, current_player = self.game.next_state(board, current_player, action)
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.terminal_value(board, 1)))
            self.display(board)
        self.game_lengths.append(it)
        return self.game.terminal_value(board, 1)

    def play_games(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.play_games', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)
        self.game_lengths = []

        num = int(num/2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in range(num):
            game_result = self.play_game(verbose=verbose)
            if game_result==1:
                one_won+=1
            elif game_result==-1:
                two_won+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = f'({eps}/{maxeps}) | Eps Time: {eps_time.avg:.3f}s | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | One first: {one_won}:{two_won} ({draws} draws)'
            bar.next()

        self.player1, self.player2 = self.player2, self.player1
        one_won_first = one_won
        two_won_second = two_won
        draws_one_first = draws

        one_won = 0
        two_won = 0
        draws = 0

        for _ in range(num):
            game_result = self.play_game(verbose=verbose)
            if game_result==-1:
                one_won+=1                
            elif game_result==1:
                two_won+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = f'({eps}/{maxeps}) Eps Time: {eps_time.avg:.3f}s | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | One fist: {one_won_first}:{two_won_second} ({draws_one_first} draws) | Two first: {two_won}:{one_won} ({draws} draws)'
            bar.next()
            
        bar.finish()
        print(f"Arena game lengths,        min:{np.min(self.game_lengths):0.0f}, avg:{np.average(self.game_lengths):0.2f}, max:{np.max(self.game_lengths):0.0f}, std:{np.std(self.game_lengths):0.2f}")
        
        return (one_won_first, one_won), (two_won, two_won_second), (draws_one_first, draws)

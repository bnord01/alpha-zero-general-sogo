from Player import Player
from Game import Game

def pit(p1: Player, p2: Player, g: Game):
  board = g.init_board()
  player = 1
  p1_moves = []
  p2_moves = []
  while not g.getTerminal(board):
    p = p1 if player == 1 else p2
    moves = p1_moves if player == 1 else p2_moves
    reason, move = p.make_move(board)
    nboard, nplayer = g.next_state(board, player, move)
    moves.append((board, reason, move))
    board, player = nboard, nplayer
  p1_result = g.terminal_value(board, 1)
  p1.episode_done(p1_moves, p1_result)
  p2_result = g.terminal_value(board, -1)
  p2.episode_done(p2_moves, p2_result)

    
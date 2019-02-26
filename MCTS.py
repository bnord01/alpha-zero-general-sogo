import math
import numpy as np
from Game import Game
from NeuralNet import NeuralNet
from typing import List


class Node(object):

    def __init__(self, prior: float = 1., player=1,  visits=0):
        self.visit_count = visits
        self.player = player # the Player who played this node
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def print(self, limit=-1, indent='', action='',filter=None):
        if limit == 0 or self.visit_count <= 0:
            return
        print(f"{indent}{action} -> v:{float(self.value()):1.2} n:{self.visit_count} p:{float(self.prior):1.2} tp:{self.player}")
        for a, child in self.children.items():
            if filter == None:
                child.print(limit=limit-1, indent=indent+'  ', action=str(a))            
            elif a in filter:
                if isinstance(filter, list):
                    child.print(limit=1, indent=indent+'  ', action=str(a))
                else:
                    child.print(limit=limit-1, indent=indent+'  ', action=str(a), filter=filter[a])
                    
        


class Play(object):

    def __init__(self, game: Game, board: np.ndarray, player=1):
        self.game = game
        self.board = board
        self.player = player

    def terminal(self):
        return self.game.getTerminal(self.board)

    def terminal_value(self):
        return self.game.getGameEnded(self.board, self.player)

    def legal_actions(self):
        return self.game.getValidMoves(self.board, self.player)

    def clone(self):
        return Play(self.game, self.board.copy(), self.player)

    def apply(self, action):
        self.board, self.player = self.game.getNextState(self.board, self.player, action)

    def canonical_board(self):
        return self.game.getCanonicalForm(self.board, self.player)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: Game, nnet: NeuralNet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

    def getActionProb(self, canonicalBoard):
        return self.get_action_prob(canonicalBoard)[0]

    def get_action_prob(self, board, player=1, root=None):

        play = Play(self.game, board, player=player)
        root = root or Node(player=-player, visits = 1)
        
        if not root.expanded():
            self.evaluate(root, play)

        self.add_exploration_noise(root)

        for _ in range(self.args.num_mcts_sims - root.visit_count):
            node = root
            scratch_play = play.clone()
            search_path = [node]
            while node.expanded():
                action, node = self.select_child(node)
                scratch_play.apply(action)
                search_path.append(node)
            
            value = self.evaluate(node, scratch_play)
            self.backpropagate(search_path, value, scratch_play.player)
                        
        return [root.children[a].visit_count/(root.visit_count - 1) if a in root.children else 0 for a in range(self.game.getActionSize())], root

    # Select the child with the highest UCB score.

    def select_child(self, node: Node):
        _, action, child = max((self.ucb_score(node, child), action, child)
                               for action, child in node.children.items())
        return action, child

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.

    def ucb_score(self, parent: Node, child: Node):
        pb_c = math.log((parent.visit_count + self.args.pb_c_base + 1) /
                        self.args.pb_c_base) + self.args.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = (child.value() + 1)/2
        return prior_score + value_score

    # We use the neural network to obtain a value and policy prediction.
    def evaluate(self, node: Node, play: Play):
        """
        Expands the `node`, initialising the children with the prior 
        and sets the player of `node` to the previous player who played
        `node`. 
        
        Returns:
            The (estimated) value of the game for the player who will play from `node`.
        """        

        if play.terminal():
            return play.terminal_value()

        pi, value = self.nnet.predict(play.canonical_board())

        # Expand the node.
        legal = play.legal_actions()
        pi = pi*legal
        pi = pi / sum(pi)
        for action in range(len(pi)):
            if legal[action] == 1:
                node.children[action] = Node(pi[action], play.player)
        return value

    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.

    def backpropagate(self, search_path: List[Node], value: float, player_to_value):
        for node in search_path:
            node.value_sum += value * node.player * player_to_value
            node.visit_count += 1

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.

    def add_exploration_noise(self, node: Node):
        actions = node.children.keys()
        noise = np.random.gamma(
            self.args.root_dirichlet_alpha, 1, len(actions))
        noise = noise/sum(noise)
        frac = self.args.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * \
                (1 - frac) + n * frac

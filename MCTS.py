import math
import numpy as np
from Game import Game
from NeuralNet import NeuralNet
from typing import List


class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Play(object):

    def __init__(self, game: Game, board: np.ndarray, actions=None):
        self.actions = actions or []
        self.child_visits = []
        self.game = game
        self.board = board
        self.num_actions = self.game.getActionSize()

    def terminal(self):
        return self.game.getTerminal(self.board)

    def terminal_value(self, to_play):
        return self.game.getGameEnded(self.board, to_play)

    def legal_actions(self):
        return self.game.getValidMoves(self.board, self.to_play())

    def clone(self):
        return Play(self.game, self.board.copy(), list(self.actions))

    def apply(self, action):
        self.actions.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(
            child.visit_count for child in root.children.itervalues())
        self.child_visits.append([
            root.children[a].visit_count /
            sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def to_play(self):
        return (-1) ** len(self.actions)

    def canonical_board(self):
        return self.game.getCanonicalForm(self.board, self.to_play())


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: Game, nnet: NeuralNet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

    def getActionProb(self, canonicalBoard, temp=1):
        root = Node(0)
        play = Play(self.game, canonicalBoard)

        self.evaluate(root, play)
        self.add_exploration_noise(root)

        for _ in range(self.args.numMCTSSims):
            node = root
            scratch_play = play.clone()
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                scratch_play.apply(action)
                search_path.append(node)

            value = self.evaluate(node, scratch_play)
            self.backpropagate(search_path, value, scratch_play.to_play())
        return self.select_action(play, root), root

    def select_action(self, play: Play, root: Node):
        visit_counts = [(child.visit_count, action)
                        for action, child in root.children.iteritems()]
        if len(play.actions) < self.args.tempThreshold:
            _, action = (0, 0)
        else:
            _, action = max(visit_counts)
        return action

    # Select the child with the highest UCB score.

    def select_child(self, node: Node):
        _, action, child = max((self.ucb_score(node, child), action, child)
                               for action, child in node.children.iteritems())
        return action, child

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.

    def ucb_score(self, parent: Node, child: Node):
        pb_c = math.log((parent.visit_count + self.args.pb_c_base + 1) /
                        self.args.pb_c_base) + self.args.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()
        return prior_score + value_score

    # We use the neural network to obtain a value and policy prediction.
    def evaluate(self, node: Node, play: Play):
        if play.terminal() :
            return play.terminal_value(1)

        pi, value = self.nnet.predict(play.canonical_board())

        # Expand the node.
        node.to_play = play.to_play()
        policy = {a: math.exp(pi[a]) for a in play.legal_actions()}
        policy_sum = sum(policy.itervalues())
        for action, p in policy.iteritems():
            node.children[action] = Node(p / policy_sum)
        return value

    # At the end of a simulation, we propagate the evaluation all the way up the
    # tree to the root.

    def backpropagate(self, search_path: List[Node], value: float, to_play):
        for node in search_path:
            node.value_sum += value * node.to_play * to_play
            node.visit_count += 1

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.

    def add_exploration_noise(self, node: Node):
        actions = node.children.keys()
        noise = np.random.gamma(0.3, 1, len(actions))
        frac = 0.25
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * \
                (1 - frac) + n * frac

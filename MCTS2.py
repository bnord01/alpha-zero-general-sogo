import math
import numpy as np
EPS = 1e-8

class MCTS2():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.terminal_value ended for board s
        self.Vs = {}        # stores game.valid_actions for board s

    def get_action_prob(self, canonical_board):
        """
        This function performs num_mcts_sims simulations of MCTS starting from
        canonical_board.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.args.num_mcts_sims):
            self.search(canonical_board)

        s = self.game.string_representation(canonical_board)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.action_size())]
      
        probs = [x/float(sum(counts)) for x in counts]
        return probs, None


    def search(self, canonical_board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonical_board
        """

        s = self.game.string_representation(canonical_board)

        if s not in self.Es:
            self.Es[s] = self.game.terminal_value(canonical_board, 1)
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonical_board)
            valids = self.game.valid_actions(canonical_board, 1)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                # print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]

        _ , a = max((self.ucb_score(s,action), action) for action in range(self.game.action_size()) if valids[action])

        next_s, next_player = self.game.next_state(canonical_board, 1, a)
        next_s = self.game.canonical_board(next_s, next_player)

        v = self.search(next_s)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v

    def ucb_score(self, s, a):
        pb_c = math.log((self.Ns[s] + self.args.pb_c_base + 1) / self.args.pb_c_base) + self.args.pb_c_init
        pb_c *= math.sqrt(self.Ns[s]) 
        if (s,a) in self.Qsa:
            return self.Qsa[(s,a)] + pb_c / (self.Nsa[(s,a)] + 1) * self.Ps[s][a]
        return pb_c * self.Ps[s][a]
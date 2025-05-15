# local import 
from .game import DotsAndBoxesGame


class AZNode:
    """
    Implements the search tree of AlphaZero. Each node corresponds to a position s.

    Attributes
    ----------
    s : DotsAndBoxesGame
        position corresponding to the node (more accurate: s is the game stat which includes the position vector s)
    a : int
        move that was executed at the parent's position, resulting in this node's position s
    children : [AZNode]
        child nodes
    Q : dict
        action values Q[a] = Q(s,a)
    N : dict
        visit counts N[a] = N(s,a)
    W : dict
        total value W[a] = sum of value backups for action a
    P : np.ndarray
        action prior probabilities P(s,a) as returned by the neural network
    N_rave : dict
        RAVE visit counts for action a
    W_rave : dict
        RAVE total values for action a
    """

    def __init__(self, parent, a: int, s: DotsAndBoxesGame):
        # any node except root needs to have a corresponding move
        assert (parent is None and a is None) or (parent is not None and isinstance(a, int))

        # if not root, link to parent
        if parent is not None:
            assert isinstance(parent, AZNode)
            parent.children.append(self)

        # assign move and state
        self.a = a
        self.s = s

        # initialize child containers
        self.children = []
        # action statistics
        self.Q = {}
        self.W = {}           # total value sums for each action
        self.N = {}
        self.P = None
        # RAVE statistics
        self.N_rave = {}
        self.W_rave = {}

    def get_child_by_move(self, a: int):
        for child in self.children:
            if child.a == a:
                return child
        return None

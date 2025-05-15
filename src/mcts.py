import copy
import math
from random import randint
import numpy as np

# local imports
from .game import DotsAndBoxesGame
from .model.neural_network import AZNeuralNetwork
from .node import AZNode


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) implementation combining:
      - PUCT with dynamic exploration constant
      - Rapid Action Value Estimate (RAVE) mixing
      - Virtual loss for safe parallelization

    Attributes:
        model (AZNeuralNetwork): Neural network providing policy priors P(s,a) and value v(s).
        root (AZNode): Root of the search tree, corresponding to the current game state.
        n_simulations (int): Number of MCTS simulations per move.
        c_puct (float): Base exploration coefficient for PUCT.
        base_c (float): Base PUCT constant (aliased to c_puct).
        alpha (float): Strength of dynamic adjustment for exploration.
        dirichlet_eps (float): Epsilon weight for Dirichlet noise at root.
        dirichlet_alpha (float): Alpha parameter for Dirichlet distribution.

    Class Constants:
        VIRTUAL_LOSS (float): Amount to subtract/add for virtual loss mechanism.
        RAVE_BETA_COEFF (float): RAVE blending coefficient (b in β = b/(b+N)).
    """

    VIRTUAL_LOSS = 1.0
    RAVE_BETA_COEFF = 1000.0

    def __init__(
        self,
        model: AZNeuralNetwork,
        s: DotsAndBoxesGame,
        mcts_parameters: dict
    ):
        # Store neural network and initialize root node
        self.model = model
        self.root = AZNode(parent=None, a=None, s=s)
        # Ensure RAVE stats containers exist on the root
        self.root.N_rave = {}
        self.root.W_rave = {}

        # Extract MCTS hyperparameters
        self.n_simulations   = mcts_parameters["n_simulations"]
        self.c_puct          = mcts_parameters.get("c_puct", 1.0)
        self.base_c          = self.c_puct
        self.alpha           = mcts_parameters.get("c_puct_alpha", 0.0)

        self.dirichlet_eps   = mcts_parameters.get("dirichlet_eps", 0.25)
        self.dirichlet_alpha = mcts_parameters.get("dirichlet_alpha", 0.03)

    def play(self, temp: int) -> [float]:
        """
        Run full MCTS for the root state and return move probabilities.
        temp: Temperature for final distribution (0 => argmax).
        """
        s = self.root.s
        valid_moves = s.get_valid_moves()

        # Perform n_simulations playouts
        for _ in range(self.n_simulations):
            # Add Dirichlet noise only at the root
            dir_noise = np.zeros(s.N_LINES, dtype=np.float32)
            dir_noise[valid_moves] = np.random.dirichlet([
                self.dirichlet_alpha] * len(valid_moves)
            )
            self.search(self.root, is_root=True, dirichlet_noise=dir_noise)

        # Compute visit counts, then temperature-scaled probabilities
        counts = [self.root.N.get(a, 0) for a in range(s.N_LINES)]
        if temp == 0:
            # Deterministic: pick the most visited move
            probs = [0] * len(counts)
            best = int(np.argmax(counts))
            probs[best] = 1
            return probs

        # Softmax with temperature (via counts^(1/temp))
        scaled = [c ** (1.0 / temp) for c in counts]
        total = sum(scaled)
        return [x / total for x in scaled]

    def dynamic_c(self, node: AZNode) -> float:
        """
        Compute a dynamic exploration coefficient:
        base_c * (1 + alpha / (1 + total_visits)).
        total_visits: sum of N(s,a) for all a at this node.
        """
        total_visits = sum(node.N.values())
        return self.base_c * (1 + self.alpha / (1 + total_visits))

    def search(
        self,
        node: AZNode,
        is_root: bool = False,
        dirichlet_noise: np.ndarray = None
    ) -> float:
        """
        Single MCTS simulation: Selection -> Expansion/Evaluation -> Backup.
        Returns: the value v ∈ [-1,1] from the leaf evaluation.
        """
        # Terminal check
        if not node.s.is_running():
            res = node.s.result
            if res == node.s.current_player:
                return 1
            elif res == 0:
                return 0
            else:
                return -1

        # If leaf (no policy), evaluate network to set P and return v
        if node.P is None:
            return self.evaluate(node)

        # Selection + Expansion
        leaf, path = self._select(node, is_root, dirichlet_noise)

        # If newly expanded leaf, network evaluated in evaluate(); else terminal
        if leaf.P is None:
            v = self.evaluate(leaf)
        else:
            res = leaf.s.result
            v = 1 if res == leaf.s.current_player else 0 if res == 0 else -1

        # Backpropagate value and undo virtual losses
        self._backup(path, v)
        return v

    def _select(
        self,
        node: AZNode,
        is_root: bool,
        dirichlet_noise: np.ndarray
    ):
        """
        Traverse from node to leaf by selecting actions that maximize:
          Q_hat + U,
        where Q_hat mixes standard Q and RAVE estimate, and
        U = dynamic_c * P * sqrt(sumN) / (1+N).
        """
        path = []
        # Continue until leaf or terminal
        while node.P is not None and node.s.is_running():
            total_N = sum(node.N.values())
            sqrt_N  = math.sqrt(total_N)
            # Apply Dirichlet noise only at root
            P = (1 - self.dirichlet_eps) * node.P + self.dirichlet_eps * dirichlet_noise if is_root else node.P

            # Select best action a
            best_score, best_action = -float('inf'), None
            for a in node.s.get_valid_moves():
                q = node.Q.get(a, 0.0)
                n = node.N.get(a, 0)
                # RAVE: rapid action value estimate
                n_rave = node.N_rave.get(a, 0)
                q_rave = (node.W_rave.get(a, 0.0) / n_rave) if n_rave > 0 else 0.0
                beta = self.RAVE_BETA_COEFF / (self.RAVE_BETA_COEFF + total_N)
                q_hat = (1 - beta) * q + beta * q_rave

                # Exploration bonus
                u = self.dynamic_c(node) * P[a] * sqrt_N / (1 + n)
                score = q_hat + u
                if score > best_score:
                    best_score, best_action = score, a

            # Virtual loss for parallel safety
            node.N[best_action] = node.N.get(best_action, 0) + self.VIRTUAL_LOSS
            node.W[best_action] = node.W.get(best_action, 0.0) - self.VIRTUAL_LOSS
            path.append((node, best_action))

            # Move to next node: expand if necessary
            child = node.get_child_by_move(best_action)
            if child is None:
                node = self.expand(node, best_action)
            else:
                node = child
            # Only apply noise on first move
            is_root, dirichlet_noise = False, None

        return node, path

    def expand(self, node: AZNode, a: int) -> AZNode:
        """
        Expand the tree by creating a child for action a.
        Copies state, applies move, initializes new AZNode.
        Child inherits RAVE containers.
        """
        s_copy = copy.deepcopy(node.s)
        s_copy.execute_move(a)
        child = AZNode(parent=node, a=a, s=s_copy)
        # Initialize RAVE stats on new child
        child.N_rave = {}
        child.W_rave = {}
        return child

    def evaluate(self, leaf: AZNode) -> float:
        """
        Query neural network for policy and value on leaf state.
        Applies random symmetries for data augmentation.
        Sets leaf.P and returns scalar v.
        """
        lines = leaf.s.get_canonical_lines()
        boxes = leaf.s.get_canonical_boxes()
        i = randint(0, 7)
        j = {1: 3, 3: 1}.get(i, i)
        rot_l = DotsAndBoxesGame.get_rotations_and_reflections_lines(lines)[i]
        rot_b = DotsAndBoxesGame.get_rotations_and_reflections_boxes(boxes)[i]
        p, v = self.model.p_v(rot_l, rot_b)
        leaf.P = DotsAndBoxesGame.get_rotations_and_reflections_lines(p)[j]
        return v

    def _backup(self, path: list, value: float):
        """
        Backpropagate value through visited path nodes, undo virtual loss,
        update Q and N for each action, and refresh RAVE stats.
        """
        for node, a in reversed(path):
            # Undo virtual loss
            node.N[a] -= self.VIRTUAL_LOSS
            node.W[a] += self.VIRTUAL_LOSS
            # Update visit count and action value Q
            prev_n = node.N.get(a, 0)
            prev_q = node.Q.get(a, 0.0)
            node.N[a] = prev_n + 1
            node.Q[a] = (prev_n * prev_q + value) / node.N[a]
            # RAVE update: all other moves in path
            for ancestor, move in path:
                if move != a:
                    ancestor.N_rave[a] = ancestor.N_rave.get(a, 0) + 1
                    ancestor.W_rave[a] = ancestor.W_rave.get(a, 0.0) + value

    @staticmethod
    def determine_move(
        model: AZNeuralNetwork,
        s: DotsAndBoxesGame,
        mcts_parameters: dict
    ) -> int:
        """
        Convenience to run MCTS and return the best move index.
        """
        mcts = MCTS(model, s, mcts_parameters)
        pi = mcts.play(temp=0)
        return int(np.argmax(pi))

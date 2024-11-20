import numpy as np
from tqdm import tqdm
from graph import GraphModel
from random import shuffle
from functools import partial

class SquareLatticeIsingModel:
    def __init__(self, N: int = 60, theta: float = 0.45, p1_init: float = 1):
        r'''
        Defines a square lattice Ising model with N x N spins.

        Args:
            N (int): the number of spins in each direction.
            theta (float): coupling parameter.
        '''
        self.N = N
        self.theta = theta
        self._initialize_lattice(p1_init)  # initializes the lattice
        self.model = self._initialize_model()
        self.comb_A, self.comb_B = self._initialize_combs(self.model)

    def _initialize_lattice(self, p1_init: float) -> None:
        r'''
        Initializes the lattice with random spins.
        '''
        self.spins = np.random.choice([-1, 1], size=(self.N, self.N), p=[1 - p1_init, p1_init])

    def _initialize_model(self) -> GraphModel:
        r'''
        Initializes the graph model.
        '''
        node_potentials = {i: np.array([1, 1]) for i in range(self.N**2)}
        edge_potential = np.array([
            [np.exp(self.theta), np.exp(-self.theta)],
            [np.exp(-self.theta), np.exp(self.theta)]
        ])
        edge_potentials = {}
        for i in range(self.N):
            for j in range(self.N):
                index = i * self.N + j
                if i < self.N - 1:
                    neighbor_index = (i + 1) * self.N + j
                    edge_potentials[(index, neighbor_index)] = edge_potential
                if j < self.N - 1:
                    neighbor_index = i * self.N + (j + 1)
                    edge_potentials[(index, neighbor_index)] = edge_potential
        model = GraphModel(node_potentials, edge_potentials)
        return model

    def _initialize_combs(self, model) -> GraphModel:
        comb_A_node_potentials = {}
        comb_B_node_potentials = {}

        # Add all edge potentials if both nodes are in Comb A
        comb_A_edge_potentials = {k: v for k, v in model.edge_potentials.items()
                                  if self._is_comb_A(k[0] // self.N, k[0] % self.N)
                                  and self._is_comb_A(k[1] // self.N, k[1] % self.N)}

        # Add all edge potentials if both nodes are in Comb B
        comb_B_edge_potentials = {k: v for k, v in model.edge_potentials.items()
                                  if self._is_comb_B(k[0] // self.N, k[0] % self.N)
                                  and self._is_comb_B(k[1] // self.N, k[1] % self.N)}

        # For all nodes augment the node potentials
        for i in range(self.N**2):
            j, k = i // self.N, i % self.N
            if self._is_comb_A(j, k):
                comb_A_node_potentials[i] = np.array([1, 1])
            else:
                comb_B_node_potentials[i] = np.array([1, 1])

        # Two comb models
        comb_A = GraphModel(comb_A_node_potentials, comb_A_edge_potentials)
        comb_B = GraphModel(comb_B_node_potentials, comb_B_edge_potentials)
        self.messages = {}

        return comb_A, comb_B

    def _update_comb(self, comb='A'):
        r'''
        Updates the comb node potentials
        '''
        comb_model = self.comb_A if comb == 'A' else self.comb_B
        for i in comb_model.get_V():
            # Initialize potentials
            potential = np.array([1.0, 1.0], dtype=np.float64)
            j, k = i // self.N, i % self.N
            for neighbor in self._get_neighbors(j, k):
                if (comb == 'A' and self._is_comb_B(neighbor[0], neighbor[1])) or (
                        comb == 'B' and self._is_comb_A(neighbor[0], neighbor[1])):
                    s_j = self.spins[neighbor[0], neighbor[1]]  # -1 or 1
                    # Update potentials for s_i = -1 (index 0) and s_i = 1 (index 1)
                    potential[0] *= np.exp(self.theta * (-1) * s_j)
                    potential[1] *= np.exp(self.theta * (1) * s_j)
            # Explicitly set the updated potential
            comb_model.node_potentials[i] = potential

    def _is_comb_A(self, i: int, j: int) -> bool:
        '''
        Returns True if the node is in comb A.
        '''
        if i == 0:
            return True
        elif i == self.N - 1:
            return False
        else:
            return (j % 2 == 0)

    def _is_comb_B(self, i: int, j: int) -> bool:
        return not self._is_comb_A(i, j)

    def _node_gibbs_step(self, i, j):
        r'''
        Performs a single Gibbs sampling step. Returns the probability parameter of the spin being flipped.
        '''
        neighbors = self._get_neighbors(i, j)
        total_neighbor_spin = np.sum(self.spins[neighbors[:, 0], neighbors[:, 1]])
        p = 1 / (1 + np.exp(-2 * self.theta * total_neighbor_spin))
        next_spin = np.random.choice([-1, 1], p=[1 - p, p])
        self.spins[i, j] = next_spin
        return p

    def _block_gibbs_step(self, update_scheme):
        if update_scheme == 'iterative':
            self._block_gibbs_step_comb('A')
            self._block_gibbs_step_comb('B')
        elif update_scheme == 'random':
            steps = [partial(self._block_gibbs_step_comb, comb='A'), partial(self._block_gibbs_step_comb, comb='B')]
            shuffle(steps)
            for step in steps:
                step()
        else:
            raise NotImplementedError('Update scheme not implemented.')

    def _block_gibbs_step_comb(self, comb='A'):
        comb_model = self.comb_A if comb == 'A' else self.comb_B
        self._update_comb(comb)
        next_spins = comb_model.sample()
        for i in comb_model.get_V():
            j, k = i // self.N, i % self.N
            self.spins[j, k] = next_spins[i]  # next_spins[i] is already in {-1, 1}

    def _get_neighbors(self, i, j):
        r'''
        Returns the neighbors of a given node.
        '''
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < self.N - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < self.N - 1:
            neighbors.append((i, j + 1))
        return np.array(neighbors)

    def sample(self, iterations: int = 1000, vis_step: int = 100, sample_type: str = 'node', **kwargs):
        if sample_type == 'node':
            return self._sample_node(iterations, vis_step, **kwargs)
        elif sample_type == 'block':
            return self._sample_block(iterations, vis_step, **kwargs)

    def _sample_node(self, iterations: int, vis_step: int, **kwargs):
        r'''
        Performs Gibbs sampling node by node.
        '''
        update_scheme = kwargs.get('update_scheme', 'iterative')

        node_samples = []
        pbar = tqdm(range(iterations))
        for i in pbar:
            if update_scheme == 'iterative':
                for j in range(self.N**2):
                    p = self._node_gibbs_step(j // self.N, j % self.N)
            elif update_scheme == 'random':
                for j in list(np.random.permutation(range(self.N**2))):
                    p = self._node_gibbs_step(int(j // self.N), int(j % self.N))
            else:
                raise NotImplementedError('Update scheme not implemented.')
            pbar.set_description(f'Node Sampler, p={p:.3f}')

            if (i + 1) % vis_step == 0 or i == 0:
                node_samples.append(self.spins.copy())
        return node_samples

    def _sample_block(self, iterations: int, vis_step: int, **kwargs):
        r'''
        Performs Gibbs sampling block by block.
        '''
        update_scheme = kwargs.get('update_scheme', 'iterative')

        block_samples = []
        pbar = tqdm(range(iterations))

        for i in pbar:
            self._block_gibbs_step(update_scheme)
            pbar.set_description('Block Sampler')

            if (i + 1) % vis_step == 0:
                block_samples.append(self.spins.copy())
        return block_samples

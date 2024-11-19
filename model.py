import numpy as np
from tqdm import tqdm

class SquareLatticeIsingModel:
    def __init__(self, N: int = 60, theta: float = 0.45, p1_init: float = 0.5):
        r'''
        Defines a square lattice Ising model with N x N spins.

        Args:
            N (int): the number of spins in each direction.
            theta (float): coupling parameter.
        '''
        self.N = N
        self.theta = theta
        self._initialize_lattice(p1_init) #initializes the lattice 
        self.A, self.B = self._initialize_combs() #gets the nodes in each comb

    def _initialize_lattice(self, p1_init: float):
        r'''
        Initializes the lattice with random spins.
        '''
        self.spins = np.random.choice([-1, 1], size=(self.N, self.N), p=[1-p1_init, p1_init])
    
    def _initialize_combs(self):
        A_nodes = [[0, i] for i in range(self.N)]
        B_nodes = [[self.N-1, i] for i in range(self.N)]
        for j in range(1, self.N-1):
            for k in range(self.N):
                if k % 2 == 0:
                    A_nodes.append([j, k])
                else:
                    B_nodes.append([j, k])
        A_nodes = np.array(A_nodes)
        B_nodes = np.array(B_nodes)
        return A_nodes, B_nodes
    
    def _node_gibbs_step(self, i, j):
        r'''
        Performs a single Gibbs sampling step. Returns the probability parameter of the spin being flipped.
        '''
        neighbors = self._get_neighbors(i, j)
        p = 1/ (1 + np.exp(-2 * self.theta * np.sum(self.spins[neighbors[:, 0], neighbors[:, 1]])))
        next_spin = np.random.choice([-1, 1], p=[1-p, p])
        self.spins[i,j] = next_spin
        return p
    
    def _block_gibbs_step(self, update_scheme):
        if update_scheme == 'iterative':
            pass
        elif update_scheme == 'random':
            if np.random() > 0.5:
                pass
            else:
                pass
    
    def _get_neighbors(self, i, j):
        r'''
        Returns the neighbors of a given node.
        '''
        neighbors = set([(i, int(np.clip(j+1, 0, self.N-1))), (i, int(np.clip(j-1, 0, self.N-1))), 
                    (int(np.clip(i+1, 0, self.N-1)), j), (int(np.clip(i-1, 0, self.N-1)), j)])
        neighbors = np.array(list(neighbors))
        return neighbors

    def sample(self, iterations: int = 1000, vis_step: int = 100, sample_type: str = 'node', **kwargs):
        match sample_type:
            case 'node':
                return self._sample_node(iterations, vis_step, **kwargs)
            case 'block':  
                return self._sample_block(iterations, vis_step, **kwargs)
        
    def _sample_node(self, iterations: int, vis_step: int, **kwargs):
        r'''
        Performs Gibbs sampling node by node.
        '''
        if 'update_scheme' in kwargs:
            update_scheme = kwargs['update_scheme']

        node_samples = []
        pbar = tqdm(range(iterations))
        for i in pbar:
            if update_scheme == 'iterative':
                for j in range(self.N**2):
                    p = self._node_gibbs_step(j // self.N, j % self.N) 
            elif update_scheme == 'random':
                for j in list(np.random.permutation(self.N**2)):
                    p = self._node_gibbs_step(int(j // self.N), int(j % self.N))
            else:
                raise NotImplementedError('Update scheme not implemented.')
            pbar.set_description(f'Node Sampler, p={p:.3f}')

            if (i+1) % vis_step == 0:
                node_samples.append(self.spins.copy())
        return node_samples
    
    
    def _sample_block(self, iterations: int, vis_step: int, **kwargs):
        r'''
        Performs Gibbs sampling block by block.
        '''
        if 'update_scheme' in kwargs:
            update_scheme = kwargs['update_scheme']
        
        block_samples = []
        pbar = tqdm(range(iterations))
        for i in pbar:
            p = self._block_gibbs_step(update_scheme)
            pbar.set_description(f'Block Sampler, p={p:.3f}')

            if (i+1) % vis_step == 0:
                block_samples.append(self.spins.copy())
        return block_samples

    def belief_propogation(self):
        '''
        Performs belief propogation on the lattice.
        '''
        pass
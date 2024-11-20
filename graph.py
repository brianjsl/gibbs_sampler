import numpy as np

class GraphModel():
    ''' 
    Simple class for undirected graphical models over binary variables.
    '''

    ## Initialized by passing in:
    ##    i) dictionary mapping from nodes to node potentials (shape [2] np arrays)
    ##    ii) dictionary mapping from edges (i,j) pairs to edge potentials (2x2 np arrays)
    def __init__(self, node_potentials, edge_potentials):
        self.node_potentials = node_potentials.copy()
        self.edge_potentials = edge_potentials.copy()
        self.V = list(self.node_potentials.keys())
        self.E = list(self.edge_potentials.keys())
        self.adjacency = {node: [] for node in self.V}
        for (i, j) in self.E:
            self.adjacency[i].append(j)
            self.adjacency[j].append(i)

    def get_V(self):
        '''
        Get nodes from the graph 
        '''
        return list(self.node_potentials.keys())

    def get_E(self):
        '''
        Get Edges from the Graph 
        '''
        return list(self.edge_potentials.keys())

    def sample(self):
        # Select root node arbitrarily
        root = self.V[0]

        # Upward message passing
        messages = {}

        #calculates the message from node to parent 
        def upward_message(node, parent):
            messages_from_children = {}
            for child in self.adjacency[node]:
                if child != parent:
                    m = upward_message(child, node)
                    messages_from_children[child] = m

            m_node_to_parent = np.zeros(2)
            for x_parent in [0, 1]:
                sum_over_x_node = 0.0
                for x_node in [0, 1]:
                    prod_messages = 1.0
                    for child in messages_from_children:
                        m = messages_from_children[child]
                        prod_messages *= m[x_node]
                    psi_node = self.node_potentials[node][x_node]
                    # Get edge potential between node and parent
                    if (node, parent) in self.edge_potentials:
                        psi_edge = self.edge_potentials[(node, parent)][x_node, x_parent]
                    else:
                        psi_edge = self.edge_potentials[(parent, node)][x_parent, x_node]
                    sum_over_x_node += psi_node * psi_edge * prod_messages
                m_node_to_parent[x_parent] = sum_over_x_node
            # Normalize message
            m_node_to_parent /= np.sum(m_node_to_parent)
            messages[(node, parent)] = m_node_to_parent
            return m_node_to_parent

        def upward_init():
            # Handle the case where the root has no parent
            for child in self.adjacency[root]:
                upward_message(child, root)

        upward_init()

        # Compute root marginal
        root_marginal = np.zeros(2)
        for x_root in [0, 1]:
            psi_root = self.node_potentials[root][x_root]
            prod_messages = 1.0
            for child in self.adjacency[root]:
                m = messages[(child, root)]
                prod_messages *= m[x_root]
            root_marginal[x_root] = psi_root * prod_messages
        root_marginal /= np.sum(root_marginal)

        # Sample root value (values are 0 or 1)
        x_root = np.random.choice([0, 1], p=root_marginal)
        samples = {root: x_root}

        # Downward sampling
        def sample_node(node, parent):
            x_parent = samples[parent]
            prob = np.zeros(2)
            for x_node in [0, 1]:
                prod_messages = 1.0
                for child in self.adjacency[node]:
                    if child != parent:
                        m = messages[(child, node)]
                        prod_messages *= m[x_node]
                psi_node = self.node_potentials[node][x_node]
                # Get edge potential between node and parent
                if (node, parent) in self.edge_potentials:
                    psi_edge = self.edge_potentials[(node, parent)][x_node, x_parent]
                else:
                    psi_edge = self.edge_potentials[(parent, node)][x_parent, x_node]
                prob[x_node] = psi_node * psi_edge * prod_messages
            prob /= np.sum(prob)
            x_node = np.random.choice([0, 1], p=prob)
            samples[node] = x_node
            for child in self.adjacency[node]:
                if child != parent:
                    sample_node(child, node)

        for child in self.adjacency[root]:
            sample_node(child, root)

        # Map sampled values from {0, 1} to {-1, 1}
        for node in samples:
            samples[node] = -1 if samples[node] == 0 else 1

        return samples

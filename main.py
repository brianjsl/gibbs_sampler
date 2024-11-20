from model import SquareLatticeIsingModel
import matplotlib.pyplot as plt
import os 

def sample_nodes(output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    graph = SquareLatticeIsingModel(N=60, theta=0.45, p1_init=0.5)
    node_samples, fraction = graph.sample(iterations=1000, vis_step =100, sample_type = 'node', update_scheme='random')
    fig1, ax1 = plt.subplots(2, 5, figsize=(15,6))
    fig1.suptitle('Node-by-Node Gibbs Sampler')
    for idx, (ax, node_sample) in enumerate(zip(ax1.flatten(), node_samples)):
        ax.imshow(node_sample, cmap='binary')
        ax.set_title(f'Iteration {(idx+1)*100}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'node_by_node_gibbs_sampler.png'))
    print(f'Node by Node fraction different value: {fraction}')

    graph_block = SquareLatticeIsingModel(N=60, theta=0.45, p1_init= 0.5)
    block_samples, fraction = graph_block.sample(iterations=1000, vis_step=100, sample_type = 'block', update_scheme='random')
    fig2, ax2 = plt.subplots(2, 5, figsize=(15,6))
    fig2.suptitle('Block Gibbs Sampler')
    for idx, (ax, node_sample) in enumerate(zip(ax2.flatten(), block_samples)):
        ax.imshow(node_sample, cmap='binary')
        ax.set_title(f'Iteration {(idx+1)*100}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'block_gibbs_sampler.png'))
    print(f'Block fraction different value: {fraction}')

if __name__ == '__main__':
    sample_nodes()
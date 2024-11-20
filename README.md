# Gibbs Sampling

An implementation of Gibbs Sampling on a Square Lattice Ising model. 
![A 60x60 spin lattice sampled with Blocked Gibbs Sampling](
    outputs/block_gibbs_sampler.png
)

The result of running `main.py` with coupling parameter `theta=0.45` and `p_init=0.5`.

## Basic Usage

The code for the implementation is mainly split into two parts:
* `model.py`: defines the `SquareLatticeIsingModel` class that creates an `N × N` 
square lattice Ising model and runs Gibbs-sampling (or the block variant) by running the
`sample` method with the requisite parameter `sample_type`. 
* `graph.py`: defines a graphical model `GraphModel` class that has a `sample` method 
that uses belief propogation to sample from the graphical model.

### Examples

An example of both blocked and node-by-node sampling are given in `main.py` in 
`sample_nodes`. The basic idea is to define a `SquareLatticeIsingModel` graph from `model`
and then run the `sample` method to get samples using Gibbs Sampling. Changing the
`sample_type` parameter of the graph you defined to `block` allows you to switch to blocked
Gibbs Sampling using a comb.

```
graph = SquareLatticeIsingModel(N=60, theta=0.45, p1_init=0.5)
node_samples, fraction = graph.sample(iterations=iterations, vis_step =vis_step, sample_type = 'node', update_scheme='random')
fig1, ax1 = plt.subplots(n_rows, 5, figsize=(15,3*n_rows))
fig1.suptitle('Node-by-Node Gibbs Sampler')
for idx, (ax, node_sample) in enumerate(zip(ax1.flatten(), node_samples)):
    ax.imshow(node_sample, cmap='gray')
    ax.set_title(f'Iteration {(idx+1)*100}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'node_by_node_gibbs_sampler.png'))
print(f'Node by Node fraction different value: {fraction}')
```



## Scoring

For user $u$ and item $i$ we compute:

$$
\text{score}(u, i) = \alpha \cdot \mathbf{u}^\top \mathbf{v}_i + (1 - \alpha) \cdot \mathrm{pop}(i)
$$

where $\mathbf{u}, \mathbf{v}_i$ are embeddings and $\mathrm{pop}(i)$ is popularity from interactions.

## Other demos

- `minimal_demo.py` — same model, simpler setup
- `pickle_demo.py` — model from `joblib`
- `matrix_demo.py` — precomputed score lookup

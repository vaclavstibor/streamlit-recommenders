## Appendix: scoring method

The demo blends a user embedding score with global popularity:

$$
\mathrm{score}(u, i) =
\alpha \cdot \mathbf{u}^{\top}\mathbf{v}_i
+ (1 - \alpha) \cdot \mathrm{pop}(i)
$$

The session profile is interactive: clicked items are passed back to the model as
`session_items`, so the same script works for dataset users and for the
`Try yourself (session)` user.

## What belongs here

- model details that do not fit in a short paper,
- parameter notes,
- extra diagnostics,
- qualitative observations from a demo session.

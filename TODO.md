# TODO

## Ideas not yet built

- Animated morph: a slider that interpolates continuously between the source
  and the target coordinate system, so the grid is seen deforming rather than
  only before and after.
- Export: PNG/SVG download beyond Plotly's own camera button, and CSV of the
  sampled points.
- Implicit curves `f(x, y) = 0` via marching squares.
- Vector fields and their images under a change of coordinates.
- Animate a parameter slider (play/pause) to sweep a family of curves.

## Known rough edges

- A complex map at high grid resolution with three scene views is the heaviest
  request in the app (roughly 1 MB). The domain-colouring view is cheap; the
  modulus/phase surfaces are what cost.
- `aspectmode: 'data'` can frame a very flat surface awkwardly on first load;
  the scene is draggable, so it is a first-impression problem only.

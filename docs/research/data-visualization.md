# Data visualization standard

QuantCredit figures make aggregate evidence easier to inspect. They do not
decorate a notebook or become a second owner of research conclusions.

## Reference

The review checklist adapts Caylent's
[`tufte-data-viz`](https://github.com/caylent/tufte-data-viz) at revision
[`ae7ca0d`](https://github.com/caylent/tufte-data-viz/tree/ae7ca0de7819db83241b24a2618810d5f1171145),
licensed MIT. It is a design reference, not a dependency, vendored source,
submodule, or installed agent skill. QuantCredit owns every local decision and
line of plotting code.

## Before drawing

Name three facts:

1. the finding or invariant the figure should reveal;
2. the comparison that makes it meaningful;
3. why a figure communicates the evidence better than prose or a table.

If the third answer is weak, do not add a figure.

## Default review

- Lead with the finding or invariant; panel titles may name subordinate
  diagnostics.
- Remove top and right spines. Use faint horizontal guides only when values need
  quantitative comparison.
- Label series directly when that is clearer than a legend.
- Use color for semantic emphasis, never as the only distinguishing channel.
- Keep shared scales across comparable small multiples.
- Start magnitude bars and probability axes at zero. Never exaggerate a small
  difference by truncating the scale.
- Avoid dual axes, 3D, pie charts, gradients, dense point markers, and rotated
  labels.
- Annotate the notable value or boundary and format numbers at meaningful
  precision.
- Keep aggregate tables available as the textual alternative to every canonical
  figure.

## QuantCredit adaptations

Sapphire dark mode and monospace typography are intentional project identity,
not defaults inherited from the reference. Its semantic colors may exceed four
when credit states genuinely differ, provided labels or patterns preserve the
meaning without color. A compact legend is acceptable when direct labels would
crowd grouped categorical evidence. Heatmap cell boundaries remain because they
encode a matrix, not a decorative panel border.

The upstream serif, off-white, zero-gridline, and legend-free rules are
therefore not copied mechanically. The invariant is comprehension without
distortion, not visual conformity.

# Plate layout, controls, and edge effects

A policy hands you 96 compounds. Where each one goes on the physical plate is
not a formatting detail -- it can introduce a bias larger than the effect you
are trying to measure, and the bias can be perfectly correlated with the
model's predictions.

## Edge effects: what they are

A microplate's outer ring of wells behaves differently from its interior.

- **Evaporation.** Edge wells have more exposed surface relative to their
  volume and sit against the plate's walls, so they lose solvent faster. The
  remaining contents concentrate -- compound, substrate and salts all become
  more concentrated than intended.
- **Temperature.** The interior of a stacked plate equilibrates more slowly
  than its edges. During an incubation, edge wells spend more time at a
  different temperature.
- **Gas exchange and humidity.** In cell-based assays, edge wells exchange
  gases differently, and cells there grow differently.

The resulting bias is commonly 5-20% of signal and is well documented in the
screening quality-control literature (Lundholt et al., *J Biomol Screen* 2003;
Caraus et al., *Brief Bioinform* 2015). It is systematic, not random, so
replicates in edge wells do not average it away.

## Why this interacts with active learning specifically

Suppose your policy returns its 96 picks in score order -- which every policy
here does -- and you fill the plate row-wise, as a liquid handler does by
default. Then:

- Row A (an edge row) gets picks 1-12: the highest-scoring compounds.
- Row H (the other edge row) gets picks 85-96: the lowest.

Now the edge bias is **perfectly correlated with the model's prediction.** Two
ways this ruins the campaign:

- If the bias is positive, your top predictions all read high. You "confirm"
  the model, feed the confirmation back in, and chase an artifact for months.
  The model becomes more confident about exactly the wrong thing.
- If negative, your best series reads flat and you discard it.

This is worse in an active-learning loop than in a one-shot screen, because the
biased measurements become training data. A one-shot screen gives you a bad
plate; a loop gives you a bad plate *and* a model that has learned the plate's
geometry.

Loopwise's own naive-layout mode reproduces this on demand, which is how the
test suite proves the stratified layout fixes it:

```python
naive = PlateLayout.assign(ids, predicted_scores=scores, randomize=False)
naive.diagnostics["top_stratum_edge_enrichment"]   # 2.0 -- twice chance
```

## What Loopwise does

### 1. Reserved control wells

`PlateTemplate` holds explicit positive- and negative-control wells, and
**nothing can overwrite them.** This is a hard invariant: asserted inline in
`PlateLayout.assign`, checked by `control_wells_intact()`, and tested across
templates, seeds and replicate counts. Every plate in a multi-plate batch gets
a full set, not just the first.

```python
from loopwise.plate import PLATE_96, PlateTemplate

PLATE_96.describe()
# 96-well (8x12 = 96 wells): 8 positive + 8 negative control wells,
# edge ring in use, 80 wells available for samples.
```

The default 96-well layout puts negative (vehicle/DMSO) controls in column 1
and positive controls in column 12. That is the bench convention, and it has a
useful property: both control columns are themselves on edges, so the controls
*sample* the edge effect and can be used to estimate it. The alternative,
`control_columns=False`, scatters controls through the interior, which measures
within-plate spatial drift better at the cost of breaking convention.

A control silently overwritten by a sample produces a plate you cannot
interpret and will not know you cannot interpret. Hence the hard invariant.

### 2. Stratified randomization

Compounds are bucketed into strata by predicted score, then assigned so each
stratum is spread across edge and interior wells and across all four quadrants
in proportion to how many such wells exist. Randomization *within* that
constraint keeps the assignment unbiased; the stratification stops randomness
from clustering the predicted actives by chance.

The mechanism (`PlateLayout._interleave`) is worth knowing because the obvious
implementation is wrong. Dealing one compound per (plate, edge, quadrant) cell
per turn -- plain round-robin -- drains the small cells first: a 96-well plate
with control columns has 20 edge sample wells against 60 interior ones, so by
the time the last stratum is dealt, the edge cells are empty and that stratum
gets *no* edge wells at all. That is still a score/position correlation, just
inverted. Loopwise instead spaces each cell's wells evenly through a global
sequence (the j-th of a cell's `c` wells gets position `(j + 0.5) / c`, then
everything is sorted by position), and does the same for the strata, so zipping
the two sequences balances both. It is Bresenham's line algorithm, applied to
plate wells.

### 3. Replicates in different positions

Replicates are placed in separate passes over the global well sequence, so a
compound's replicates land in different cells and usually on different plates.
Replicates sharing a position bias give a reassuringly tight standard deviation
around a systematically wrong mean, which is worse than no replicates at all --
it converts a known unknown into a false certainty.

### 4. An auditable diagnostic

Every layout reports edge enrichment per score stratum: the fraction of that
stratum's wells that landed on the edge, divided by the fraction of available
sample wells that are edge wells. 1.0 means chance; 2.0 means twice as
edge-exposed as it should be.

```python
layout = PlateLayout.assign(ids, predicted_scores=scores, seed=0)
print(layout.summary())
```

```
96-well (8x12 = 96 wells): 8 positive + 8 negative control wells, edge ring in
use, 80 wells available for samples.
80 sample wells across 1 plate(s); controls intact: True.
Edge wells are 25% of available sample wells; highest-score stratum edge
enrichment vs chance 0.75 (balanced). With 80 samples in 4 strata this ratio is
coarse -- one well either way moves it noticeably.
```

That last sentence matters: with 20 edge wells and 4 strata, chance gives each
stratum 5 wells, so a single well moves the ratio by 0.2. Treat the number as
"not systematically clustered", not "uniform to three decimals".

## Outputs

```python
layout.to_csv("plate_map.csv")              # one row per well
layout.to_grid(plate=0, value="item_id")    # how the plate looks on the bench
render_plate_map(layout, "plate_map.png")   # the figure
```

The figure shades sample wells by predicted score and draws edge wells with a
heavy outline, because the whole point of looking at it is to check that the
dark wells have not clustered against the outline.

The CSV columns are `plate, well, row, column, content_type, item_id,
replicate, predicted_score, score_stratum, is_edge`. `content_type` is one of
`sample`, `positive_control`, `negative_control`, `empty`.

## The 384-well case

```python
PlateTemplate.plate384()  # 16 x 24, outer ring excluded by default
```

`exclude_edge_wells` defaults to **True** for 384-well plates and **False** for
96-well. The reasoning: the ring is 20% of a 384-well plate against 36% of a
96, so sacrificing it is far cheaper; and 384-well edge effects are worse,
because smaller wells have a higher surface-to-volume ratio and therefore
evaporate proportionally faster. Leaving the ring empty is the most
conservative answer available, and on a 384 it is affordable.

## Choosing a layout

| Situation | Setting |
| --- | --- |
| Standard 96-well primary screen | `PLATE_96` (controls in columns 1 and 12) |
| 384-well screen | `PlateTemplate.plate384()` -- ring excluded |
| Cell-based assay, edge effects known to be severe | `plate96(exclude_edge_wells=True)`; costs 34 of 96 wells |
| You need to estimate within-plate spatial drift | `plate96(control_columns=False)` -- interior checkerboard |
| Comparing to a historical plate layout | pass explicit `positive_control_wells` / `negative_control_wells` |

## What Loopwise does not do

- **It does not model the edge effect**, only avoid confounding it. If you want
  to *correct* for spatial bias, fit a plate-position model to your control
  wells; the `is_edge`, `row` and `column` columns in the plate map are there to
  make that straightforward.
- **It does not know your liquid handler.** The plate map is a CSV; translating
  it into a worklist is instrument-specific.
- **It does not handle compound-specific constraints** (DMSO solubility limits,
  compounds that must not be adjacent, dispense ordering for carryover). If you
  need those, generate the layout and then post-process the CSV.

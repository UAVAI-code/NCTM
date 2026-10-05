# NCTM

Python implementation of the evidence fusion core used in Normality
Calibrated Trust Measurement (NCTM). This initial release accepts three calibrated
support values and returns trust, uncertainty and a separate conflict diagnostic.

The examples use synthetic numbers. This initial release provides the fusion component. The complete research
implementation and reproduction of manuscript results are outside its current scope.

## Run

Python 3.10 or later is required. There are no third party dependencies, model
downloads or GPU requirements. Run these commands from this directory:

```bash
python3 example.py
python3 example.py --supports 0.95 0.90 0.10
python3 example.py --supports 0.95 0.90 0.10 --json
python3 -m unittest discover -s tests -v
```

## Use in Python

```python
from fusion import fuse_supports

result = fuse_supports([0.95, 0.90, 0.10])
print(result.trust, result.uncertainty, result.conflict)
```

Each input is already calibrated to the interval `[0, 1]`, with larger values
indicating stronger support for reliability. The intended order is anomaly,
perceptual quality and structural evidence. Raw detector confidences, quality
scores or anomaly distances are not interchangeable with these inputs.

## Included computation

For each support `e`, the implementation forms a binomial subjective opinion:

```text
u = u_min + (u_max - u_min) * 4 * e * (1 - e)
b = (1 - u) * e
d = (1 - u) * (1 - e)
```

The base rate is `0.5`. Defaults are `u_min = 0.05` and `u_max = 0.35`.
Averaging belief fusion assigns weights proportional to `1 / u` and gives
`U = 3 / sum(1 / u)`. Conflict `C` is the mean of the three pairwise terms
`b_i * d_j + d_i * b_j`.

Trust `T` is the harmonic mean of the three supports. An exactly zero support
produces zero trust, using the continuous boundary value of the harmonic mean.
Conflict is reported independently and does not modify `T`. The fused opinion
is also returned; its belief component or projected probability is not the
reported trust score.

## Scope of this release

Included:

- Support to opinion transformation
- Averaging belief fusion and uncertainty
- Pairwise conflict diagnostics
- Trust aggregation with the harmonic mean
- Synthetic examples and numerical checks

Not included:

- Image processing, evidence extractors or pretrained weights
- Fitting reference profiles or calibrating raw evidence scores
- Normal state stabilization or additional risk gates
- Corruption generation, benchmark data, baselines or experiment results

The scores demonstrate the fusion calculation. They are not calibrated
probabilities of downstream perception success. The full experimental pipeline
cannot be reproduced from this release alone.

## Files

```text
fusion.py             Core calculation
example.py               Synthetic examples and command line interface
tests/test_fusion.py  Input checks and mathematical properties
```

This is an initial research release. Publication metadata and a full release are
not included.

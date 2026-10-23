# Investigation after V4 scored 0.94770

## Observed public evidence

V1 0.94719; V2 0.94785; V3 0.94510; V4 time-balanced blend 0.94770.
These are user-reported Kaggle results. `leaderboard.csv` records the file hashes.
V4 was renamed to `submissions/submission_v4.csv`; it is byte-identical to the
prepared candidate, excluding an upload-file mix-up as an explanation.

The target 0.955 requires +0.00715 over V2. V4 lost 0.00015, despite improving
four overlapping historical validation windows. Historical screening is useful
for rejecting weak candidates but has not established a reliable forecast of
public-score improvement. The exact public/private labels are unavailable.
Do not interpret the small V4 delta as a statistically established degradation
without paired public labels. It is nevertheless a failed leaderboard improvement.

## Why a larger change is being tested

V4 retained 75% V2 and had test rank correlation 0.998729 with V2. It made a small
change to substantially the same learned signal. This proximity does not mathematically
preclude a large AUC improvement, but there was no evidence supporting a jump to
0.955. V3 changed rankings more, yet its local gains were period-dependent and
it scored worse publicly. Neither model complexity nor blending alone solves this.

## Graph risk propagation: rejected

There are 23,900 training edges with known labels at both endpoints: 21,772
licit→licit, 597 licit→illicit, 674 illicit→licit and 857 illicit→illicit.
These selected labelled edges show association, but are not representative proof
that unknown/test neighbours have the same labels or prediction quality.

Four fixed, label-free graph prediction transformations were tested: incoming,
outgoing, undirected probability averaging, and undirected log-odds averaging,
all with 25% neighbour weight. Neighbour probabilities come from a model trained
strictly before the validation window. No neighbour labels enter the calculation.

Every transformation lost pooled AUC on every initial window. Mean differences
were −0.00969 incoming, −0.02453 outgoing, −0.03050 undirected probabilities, and
−0.00961 undirected log-odds. No candidate advanced and no submission was created.
The finding rejects these simple propagation rules, not every graph neural model.
Full data and source snapshots: `runs/graph_risk_propagation_01/`.

## New-signal tests

`test_new_signal.py` implements two bounded candidates, with frozen parameters:

- Ordered CatBoost, 1,000 depth-6 trees, learning rate .04, L2=10. Blend 25% global
  percentile ranks with 75% V2 ranks. Ranking aligns disparate score scales;
  percentile values are **ranking scores, not calibrated illicit probabilities**.
- Confidence-screened pseudo-labels from unknown historical training rows only.
  Thresholds are checked on the last three steps inside each training fold,
  using a teacher trained strictly before that slice. Require at least 20
  calibration examples and empirical precision ≥99%; cap pseudo rows per class
  at the number of true labelled rows in that class and assign weight .2.
  Unknown/test precision cannot be observed, so this is conservative screening,
  not a guarantee of pseudo-label correctness. Blend predictions 25% with V2.

For the first fold (train through 7), no pseudo-label confidence threshold passed
screening. Even the 0.001 negative threshold had only 96.9% empirical precision
on steps 5–7; positive predictions above .99 had only nine examples and 44.4%
precision. These small historical checks caution against interpreting model
confidence as guaranteed correctness under time shift. The fold correctly uses
only real labels when no threshold qualifies.

Selection uses the same overlapping development and additional-cutoff windows as
recent work and remains adaptive to past analysis. It does not create an untouched
holdout. Results and final selection are stored in `runs/new_signal_01/`; only a
candidate satisfying the saved rule is prepared as a submission. No result is
claimed to have reached 0.955 until an actual Kaggle score is returned.

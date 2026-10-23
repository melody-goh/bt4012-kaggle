# Competition context update

Source: data description pasted by the user on 18 October 2026 (Singapore).
This is user-provided competition documentation, not independently fetched rules.

- Target is illicit probability: 1 illicit, 0 licit; unknown labels are not negatives.
- Test periods occur after training periods; temporal generalisation is required.
- The provided directed edge list may explicitly be used for engineered graph
  features or graph models. This resolves the earlier uncertainty about using
  the supplied graph for these purposes.
- Local and immediate-neighbour aggregate feature groups exist, but exact column
  identities are undisclosed. The first-94/last-71 semantic mapping is not verified.
- The description calls features `feature_1`…`feature_165`; actual CSV headers use
  `feat_1`…`feat_165`. Code follows the inspected headers.
- No permission for external labels, original Elliptic label recovery, external
  training data, or pretraining follows from this description.

Initially the user confirmed three scored submissions:
V1 0.94719, V2 0.94785, V3 0.94510. Scores are user-confirmed; no Kaggle API
submission or leaderboard access has occurred. See the diagnostic run's
`leaderboard.csv` for filenames and immutable file hashes.

Update: V4, the time-balanced blend, subsequently scored 0.94770 (user report),
0.00015 below V2. Four public results are now recorded in `leaderboard.csv`.
No Kaggle account access or submission tool has been used.

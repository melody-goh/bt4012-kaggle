from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

here=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb.cells=[
md('# BT4012 — Submission diagnostics\n\n## tl;dr\nAll three scored files reproduce to floating-point precision in the original Python 3.10 environment. V2 remains the incumbent at **0.94785 public ROC-AUC**. V3 gains concentrate in particular historical periods; its overall validation ranking did not transfer. The fixed new graph ensemble failed its advancement rule.'),
md('## Context & Methods\nPublic scores (Kaggle leaderboard): V1 0.94719, V2 0.94785, V3 0.94510. No other scored submissions exist. Reconstructed models use three 14-step windows; the middle window overlaps both others. All are retrospective diagnostics.\n\n### Key Assumptions\nUnknown labels are excluded. Provided graph features are explicitly allowed by the competition description. First-94/last-71 feature semantics are unverified. No test labels or external labels are used. Exact reproduction uses `requirements-original.txt`.\n\n### Reproduce\nRun `python diagnose_submissions.py --run-id NEW_NAME` in `experiments/` to refit the models. The summary script currently targets `runs/submission_diagnosis`; update that path for a different run. This notebook reads the saved run and recomputes the key comparison tables. Full details: `SUBMISSION_FINDINGS.md`.'),
code("from pathlib import Path\nimport json, sys\nimport pandas as pd\nfrom IPython.display import display, Image\nHERE=Path.cwd() if (Path.cwd()/'pipeline.py').exists() else Path.cwd()/'experiments'\nRUN=HERE/'runs/submission_diagnosis'\nassert (RUN/'COMPLETE').exists()\nprint('Diagnostic source:', RUN.resolve())"),
md('## Data\nThe provided train/test/edge/sample files are recorded by hash in the manifest. Public scores are from the Kaggle leaderboard; prediction reproducibility is measured separately.'),
code("display(pd.read_csv(RUN/'leaderboard.csv')[['model','filename','public_roc_auc']])\ndisplay(pd.read_csv(RUN/'submission_reproduction.csv'))\nmanifest=json.loads((RUN/'manifest.json').read_text())\ndisplay(pd.Series(manifest['versions'],name='Version').to_frame())"),
md('## Results\n### Model ranking changes with the evaluation period'),
code("metrics=pd.read_csv(RUN/'metrics.csv')\ntable=metrics[metrics.model.isin(['v1','v2','v3'])].pivot(index='model',columns='train_end',values='roc_auc')\ntable.columns=[f'Train through {x}' for x in table.columns]\ntable['Kaggle public']=pd.read_csv(RUN/'leaderboard.csv').set_index('model').public_roc_auc\ndisplay(table.round(6))\ndisplay(Image(filename=str(RUN/'local_vs_public_auc.png')))"),
md('Historical V3 gains are not uniformly transferable. Public V2 is lower than one local window and higher than another; there is no defensible fixed correction from validation to Kaggle.'),
md('### Gains concentrate in particular time steps'),
code("display(pd.read_csv(RUN/'step_win_loss_summary.csv').round(6))\ndisplay(Image(filename=str(RUN/'per_step_auc_changes.png')))\nloo=pd.read_csv(RUN/'leave_one_step_out_diagnostic.csv')\ndisplay(loo[((loo.train_end==14)&(loo.excluded_step==24)) | ((loo.train_end==21)&(loo.excluded_step==32))].round(6))"),
md('Removing step 24 flips the train-through-14 V3 advantage to a loss. This is a sensitivity diagnostic, not permission to remove difficult periods from evaluation. Pairwise AUC is not additive across steps.'),
md('### Historical uncertainty'),
code("display(pd.read_csv(RUN/'historical_bootstrap.csv').round(6))"),
md('Paired resampling of whole time steps, 1,000 draws, seed 42. All percentile intervals include zero. These intervals do not capture adjacent-step dependence or future shift and are not forecasts of Kaggle uncertainty.'),
md('### Test ranking and cross-period ordering'),
code("display(pd.read_csv(RUN/'submission_rank_correlations.csv',index_col=0).round(6))\npairs=pd.read_csv(RUN/'auc_pair_decomposition.csv')\ndisplay(pairs[pairs.model.isin(['v2','v3'])].round(6))"),
md('V3 changes the ordering much more than adding V2 graph features did. Correlation alone cannot identify correctness. Roughly 91–93% of historical positive/negative pairs cross time steps, so both pooled and per-step AUC deserve review.'),
md('### New fixed graph-only ensemble'),
code("candidate=HERE/'runs/graph_bagging_initial'\nassert (candidate/'COMPLETE').exists()\ndisplay(pd.read_csv(candidate/'metrics.csv').pivot(index='train_end',columns='model',values='roc_auc').round(6))\nprint(json.loads((candidate/'decision.json').read_text()))"),
md('The candidate improves two windows but loses on train-through-21 evaluation. It fails the frozen advancement criterion, so no submission was generated.'),
md('### Original-file preservation'),
code("sys.path.insert(0,str(HERE))\nfrom pipeline import sha256\noriginal=json.loads((HERE/'original_checksums.json').read_text())\nchanged=[name for name,digest in original.items() if sha256(HERE.parent/name)!=digest]\nassert not changed, changed\nprint(f'{len(original)} original files unchanged.')"),
md('## Takeaways\nKeep V2 as the incumbent. Require broader historical robustness before promoting a candidate; do not trust a gain driven by one period. The next planned direction is training-only feature stability selection, not a larger sweep against reused folds. Three public scores cannot ensure a 0.96+ hidden-test result. No new submission was made.')
]
nbf.validate(nb)
path=here/'Submission_Diagnostics.ipynb'
nbf.write(nb,path)
NotebookClient(nb,timeout=600,resources={'metadata':{'path':str(here)}}).execute()
nbf.write(nb,path)
print(path)

"""Create and execute only the new companion notebook."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

here = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.cells = [
    nbf.v4.new_markdown_cell('# BT4012 — Experimental pipeline\n\n## tl;dr\nThe original notebooks and submissions are preserved. This companion reviews immutable CLI runs; modeling logic lives in `pipeline.py` and `run.py`. Exact V2 reproduction must be judged from the measured differences below. No new Kaggle score is claimed.'),
    nbf.v4.new_markdown_cell('## Context & Methods\n\nDevelopment: train 1–7, validate 8–21. Later historical check: train 1–21, evaluate 22–35. Both horizons are 14 steps. Validation windows do not overlap. The later period has been used before and is **not an untouched holdout**.\n\n### Key Assumptions\nV7 uses the released graph and available same-step covariates without labels. Competition permission remains unverified because local rules were absent. Unknown labels are excluded from supervised learning. Feature order has no assumed semantic interpretation.\n\nTables are used here for exact run receipts and metric comparisons; per-step CSVs retain the detailed evidence.'),
    nbf.v4.new_markdown_cell('### 1. Setup'),
    nbf.v4.new_code_cell("from pathlib import Path\nimport json, subprocess, sys\nimport pandas as pd\nfrom IPython.display import display\nHERE = Path.cwd() if (Path.cwd() / 'run.py').exists() else Path.cwd() / 'experiments'\nassert (HERE / 'run.py').exists()\nprint('Pipeline:', HERE.resolve())"),
    nbf.v4.new_markdown_cell('### 2. Reproduce experiments\nSet `RERUN=True` to run the fixed experiment again. Each invocation creates a new output directory. Review development output before separately running `evaluate --role historical_check`. Existing saved runs are sufficient to review the recorded results.'),
    nbf.v4.new_code_cell("RERUN = False\nif RERUN:\n    for arguments in [['reproduce-v2'], ['evaluate', '--role', 'development']]:\n        subprocess.run([sys.executable, str(HERE / 'run.py'), *arguments], check=True)\nelse:\n    print('Reviewing saved completed runs. Set RERUN=True for fresh reproduction and development runs.')"),
    nbf.v4.new_markdown_cell('## Data\nThe authoritative inputs are the four supplied CSVs in the project root. Their SHA-256 hashes and library versions are saved in each run manifest.'),
    nbf.v4.new_code_cell("runs = sorted(p for p in (HERE / 'runs').iterdir() if (p / 'COMPLETE').exists())\naudit = json.loads((HERE / 'runs/v2_reproduction/audit.json').read_text())\ndisplay(pd.Series(audit, name='Observed value').to_frame())"),
    nbf.v4.new_markdown_cell('## Results\n### V2 reproduction'),
    nbf.v4.new_code_cell("comparisons = []\nfor run in runs:\n    if (run / 'reproduction.json').exists():\n        comparisons.append({'run': run.name, **json.loads((run / 'reproduction.json').read_text())})\ndisplay(pd.DataFrame(comparisons).drop(columns=['reference_sha256']))"),
    nbf.v4.new_markdown_cell('### Frozen model comparison\nPooled ROC-AUC is the competition metric. AP is a secondary diagnostic. Do not average overlapping legacy windows as independent evidence.'),
    nbf.v4.new_code_cell("tables = [pd.read_csv(run / 'metrics.csv').assign(run=run.name) for run in runs if (run / 'metrics.csv').exists()]\nresults = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()\ndisplay(results.round(6))"),
    nbf.v4.new_markdown_cell('### Time-step variation\nMacro AUC is an unweighted mean across steps with both classes; it differs from pooled AUC. The range describes historical variability, not a confidence interval.'),
    nbf.v4.new_code_cell("step_tables = [pd.read_csv(run / 'metrics_by_step.csv').assign(run=run.name) for run in runs if (run / 'metrics_by_step.csv').exists()]\nsteps = pd.concat(step_tables, ignore_index=True) if step_tables else pd.DataFrame()\nif not steps.empty:\n    display(steps.groupby(['run','model']).roc_auc.agg(['mean','min','max','count']).round(6))"),
    nbf.v4.new_markdown_cell('### Preservation check'),
    nbf.v4.new_code_cell("sys.path.insert(0, str(HERE))\nfrom pipeline import sha256\noriginal = json.loads((HERE / 'original_checksums.json').read_text())\nchanged = [name for name, expected in original.items() if sha256(HERE.parent / name) != expected]\nassert not changed, changed\nprint(f'All {len(original)} original files are byte-for-byte unchanged.')"),
    nbf.v4.new_markdown_cell('## Takeaways\nUse `AUDIT.md` for the findings and proposed next experiments and `RESULTS.md` for the measured candidate decision. No local score guarantees a private leaderboard improvement. V2 lineage remains unresolved if the comparison is not exact. Keep the preserved successful submission as the benchmark until stronger evidence justifies replacement.')
]
nbf.validate(nb)
path = here / 'Experimental_Pipeline.ipynb'
nbf.write(nb, path)
NotebookClient(nb, timeout=600, resources={'metadata': {'path': str(here)}}).execute()
nbf.write(nb, path)
print(path)

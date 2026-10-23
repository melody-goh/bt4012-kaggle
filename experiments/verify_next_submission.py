"""Replay the selected final model and verify the submission against saved artifacts."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import joblib
from pipeline import load_data,graph_features,sha256,validate_submission

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/next_submission_step_balance'


def main():
    manifest=json.loads((RUN/'manifest.json').read_text())
    for name,digest in manifest['inputs'].items():
        if sha256(HERE.parent/name)!=digest:
            raise ValueError(f'Input changed: {name}')
    original=json.loads((HERE/'original_checksums.json').read_text())
    changed=[name for name,digest in original.items() if sha256(HERE.parent/name)!=digest]
    if changed:
        raise ValueError(f'Original files changed: {changed}')
    train,test,edges,sample=load_data(HERE.parent)
    _,test,_,_=graph_features(train,test,edges)
    model=joblib.load(RUN/'final_step_balanced_graph__model.joblib')
    cols=json.loads((RUN/'final_step_balanced_graph__features.json').read_text())
    p=model.predict_proba(test[cols])[:,list(model.classes_).index(1)]
    reference=pd.read_csv(HERE.parent/'submission_v2.csv')
    candidate=pd.read_csv(RUN/'submission_next_step_balanced_graph.csv')
    validate_submission(reference,test,sample)
    validate_submission(candidate,test,sample)
    weight=manifest['config']['candidate_weight']
    reconstructed=(1-weight)*reference.target.to_numpy()+weight*p
    error=float(np.max(np.abs(reconstructed-candidate.target.to_numpy())))
    if error>1e-12:
        raise ValueError('Saved-model prediction replay mismatch')
    baseline=pd.read_csv(RUN/'metrics.csv')
    loo=pd.read_csv(RUN/'leave_one_step_out.csv')
    summary=baseline[['phase','train_end','baseline_auc','roc_auc','delta_auc']].merge(
        loo.groupby(['phase','train_end']).delta_auc.agg(leave_one_step_out_min='min',leave_one_step_out_max='max').reset_index(),
        on=['phase','train_end'],validate='one_to_one')
    summary.to_csv(RUN/'selection_summary.csv',index=False)
    qa={'rows':len(candidate),'columns':list(candidate.columns),'unique_indices':bool(candidate['index'].is_unique),
        'test_order_matches':bool(np.array_equal(candidate['index'],test['index'])),
        'finite_probabilities':bool(np.isfinite(candidate.target).all()),
        'probabilities_in_unit_interval':bool(candidate.target.between(0,1).all()),
        'saved_model_replay_max_difference':error,'original_files_checked':len(original),'original_files_changed':changed,
        'mean_absolute_probability_change_vs_v2':float((candidate.target-reference.target).abs().mean()),
        'submission_sha256':sha256(RUN/'submission_next_step_balanced_graph.csv')}
    # Single clearly named user-facing copy; fail rather than overwrite a different file.
    destination=HERE/'submissions/submission_next_step_balanced_graph.csv'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists() and sha256(destination)!=qa['submission_sha256']:
        raise FileExistsError(destination)
    if not destination.exists():
        shutil.copyfile(RUN/'submission_next_step_balanced_graph.csv',destination)
    qa['deliverable']=str(destination.resolve())
    (RUN/'verification.json').write_text(json.dumps(qa,indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(qa,indent=2))


if __name__=='__main__':
    main()

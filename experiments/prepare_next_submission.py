"""Bounded candidate comparison and conditional final submission, in original environment."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from pipeline import BASE,load_data,graph_features,fit_predict,metrics,sha256,validate_submission

HERE=Path(__file__).resolve().parent
BASE_PARAMS=dict(n_estimators=300,learning_rate=.03,num_leaves=31,random_state=42,verbosity=-1,n_jobs=4)


def fit_candidate(name,tr,va,cols,config,out,tag):
    selected=cols
    if name=='stable_features':
        # Fit feature selectors only on historical prefixes inside outer training.
        cutoffs=sorted(set(max(2,int(tr.time_step.max()*fraction)) for fraction in [.5,.75,1.]))
        rankings=[]
        for cutoff in cutoffs:
            inner=tr[tr.time_step<=cutoff]
            selector=LGBMClassifier(**BASE_PARAMS)
            selector.fit(inner[cols],inner.label.astype(int))
            gain=pd.Series(selector.booster_.feature_importance(importance_type='gain'),index=cols)
            rankings.append(gain.rank(pct=True))
        score=pd.concat(rankings,axis=1).mean(axis=1)
        selected=score.sort_values(ascending=False,kind='stable').head(config['candidates'][name]['retained_features']).index.tolist()
        # Preserve original column order for determinism.
        selected=[c for c in cols if c in selected]
        score.sort_values(ascending=False).to_csv(out/f'{tag}__stability_scores.csv',header=['mean_gain_rank'])
        model,p=fit_predict(tr,va,selected,BASE_PARAMS)
    elif name=='extra_trees':
        model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesClassifier(**config['candidates'][name]))
        model.fit(tr[cols],tr.label.astype(int))
        p=model.predict_proba(va[cols])[:,list(model.classes_).index(1)]
    elif name=='dart':
        model,p=fit_predict(tr,va,cols,config['candidates'][name])
    elif name=='step_balanced_graph':
        weights=1. / tr.groupby('time_step').label.transform('size').to_numpy()
        weights /= weights.mean()
        model=LGBMClassifier(**BASE_PARAMS)
        model.fit(tr[cols],tr.label.astype(int),sample_weight=weights)
        p=model.predict_proba(va[cols])[:,list(model.classes_).index(1)]
    else:
        raise ValueError(name)
    import joblib
    joblib.dump(model,out/f'{tag}__model.joblib',compress=3)
    (out/f'{tag}__features.json').write_text(json.dumps(selected,indent=2))
    return p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default=datetime.now(timezone.utc).strftime('next_%Y%m%dT%H%M%S%fZ'))
    parser.add_argument('--config',type=Path,default=HERE/'configs/next_submission.json')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in {'.','..'}:
        parser.error('Invalid run ID')
    out=HERE/'runs'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    cfgpath=args.config
    config=json.loads(cfgpath.read_text())
    manifest={'config':config,'python':sys.version,
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','scikit-learn','scipy','joblib']},
        'sources':{p.name:sha256(p) for p in [Path(__file__),HERE/'pipeline.py',cfgpath]},
        'inputs':{f:sha256(HERE.parent/f) for f in ['train.csv','test.csv','txs_edgelist.csv','sample_submission.csv','submission_v2.csv']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for path in [Path(__file__),HERE/'pipeline.py',cfgpath]:
        (out/('source_'+path.name)).write_bytes(path.read_bytes())
    train,test,edges,sample=load_data(HERE.parent)
    train,test,gcols,audit=graph_features(train,test,edges)
    cols=BASE+gcols
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    records=[];step_records=[];sensitivity=[]
    def evaluate(cutoff,names,phase):
        tr=train.loc[train.label.notna() & (train.time_step<=cutoff)]
        va=train.loc[train.label.notna() & train.time_step.between(cutoff+1,cutoff+14)]
        _,ref=fit_predict(tr,va,cols,BASE_PARAMS)
        result=va[['txId','time_step','label']].copy()
        result['v2']=ref
        reference_auc=roc_auc_score(va.label,ref)
        for name in names:
            p=fit_candidate(name,tr,va,cols,config,out,f'{phase}_{cutoff}_{name}')
            blend=(1-config['candidate_weight'])*ref+config['candidate_weight']*p
            result[name+'_raw']=p;result[name]=blend
            record={'phase':phase,'train_end':cutoff,'model':name,'baseline_auc':reference_auc,
                    'raw_auc':roc_auc_score(va.label,p),'delta_auc':roc_auc_score(va.label,blend)-reference_auc,
                    **metrics(va.label,blend)}
            records.append(record)
            for step in sorted(va.time_step.unique()):
                mask=va.time_step.to_numpy()==step
                step_records.append({'phase':phase,'train_end':cutoff,'time_step':int(step),'model':name,
                                     'v2_auc':roc_auc_score(va.label[mask],ref[mask]) if va.label[mask].nunique()==2 else np.nan,
                                     **metrics(va.label[mask],blend[mask])})
                keep=~mask
                sensitivity.append({'phase':phase,'train_end':cutoff,'excluded_step':int(step),'model':name,
                                    'delta_auc':roc_auc_score(va.label[keep],blend[keep])-roc_auc_score(va.label[keep],ref[keep])})
            pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
            pd.DataFrame(step_records).to_csv(out/'metrics_by_step.csv',index=False)
            pd.DataFrame(sensitivity).to_csv(out/'leave_one_step_out.csv',index=False)
            print(json.dumps(record),flush=True)
        result.to_csv(out/f'{phase}_{cutoff}_predictions.csv',index=False)
    for cutoff in config['development_origins']:
        evaluate(cutoff,list(config['candidates']),'development')
    table=pd.DataFrame(records).groupby('model').delta_auc.agg(['mean','min'])
    eligible=table[(table['mean']>0)&(table['min']>=-.0005)].sort_values(['mean','min'],ascending=False)
    chosen=eligible.index[0] if len(eligible) else None
    decision={'development_summary':table.to_dict(orient='index'),'frozen_challenger':chosen,
              'confirmation_origins':config['confirmation_origins'],'selection_rule':config['selection']}
    (out/'selection_before_confirmation.json').write_text(json.dumps(decision,indent=2))
    if chosen:
        for cutoff in config['confirmation_origins']:
            evaluate(cutoff,[chosen],'confirmation')
        check=pd.DataFrame(records).query("phase == 'confirmation'").delta_auc
        approved=bool(check.mean()>0 and check.min()>=-.001)
    else:
        approved=False
    decision['passes_confirmation']=approved
    decision['submission_model']=chosen if approved else 'v2_incumbent'
    ref=pd.read_csv(HERE.parent/'submission_v2.csv')
    validate_submission(ref,test,sample)
    if approved:
        p=fit_candidate(chosen,train.dropna(subset=['label']),test,cols,config,out,'final_'+chosen)
        submission=ref.copy()
        submission.target=(1-config['candidate_weight'])*ref.target+config['candidate_weight']*p
        file='submission_next_'+chosen+'.csv'
    else:
        submission=ref.copy()
        file='submission_incumbent_v2.csv'
    validate_submission(submission,test,sample)
    submission.to_csv(out/file,index=False)
    validate_submission(pd.read_csv(out/file),test,sample)
    decision['submission_file']=file
    decision['submission_sha256']=sha256(out/file)
    decision['max_probability_change_vs_v2']=float((submission.target-ref.target).abs().max())
    decision['rank_correlation_vs_v2']=float(submission.target.corr(ref.target,method='spearman'))
    (out/'decision.json').write_text(json.dumps(decision,indent=2))
    (out/'COMPLETE').write_text('Completed successfully\n')
    print(json.dumps(decision,indent=2),flush=True)
    print('Artifacts:',out,flush=True)


if __name__=='__main__':
    main()

"""Ordered boosting and training-only pseudo labels: bounded new-signal experiment."""
import argparse
from pathlib import Path
import importlib.metadata
import json
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pipeline import BASE,load_data,graph_features,fit_predict,metrics,sha256,validate_submission

HERE=Path(__file__).resolve().parent
PARAMS=dict(n_estimators=300,learning_rate=.03,num_leaves=31,random_state=42,verbosity=-1,n_jobs=4)


def percentile(scores):
    return rankdata(scores,method='average')/len(scores)


def pseudo_fit(train,future,cols,cutoff,settings,out,tag):
    labelled=train.loc[train.label.notna() & (train.time_step<=cutoff)]
    inner_end=cutoff-settings['calibration_steps']
    earlier=labelled[labelled.time_step<=inner_end]
    calibration=labelled[labelled.time_step>inner_end]
    _,cp=fit_predict(earlier,calibration,cols,PARAMS)
    thresholds={};audit=[]
    for target in [0,1]:
        options=settings['positive_thresholds'] if target else settings['negative_thresholds']
        for threshold in options:
            selected=cp>=threshold if target else cp<=threshold
            count=int(selected.sum())
            precision=float((calibration.label.to_numpy()[selected]==target).mean()) if count else None
            audit.append({'target':target,'threshold':threshold,'count':count,'precision':precision})
            if count>=settings['minimum_calibration_examples'] and precision>=settings['required_precision']:
                thresholds[target]=threshold
                break
    unknown=train.loc[train.label.isna() & (train.time_step<=cutoff)].copy()
    teacher,up=fit_predict(labelled,unknown,cols,PARAMS)
    batches=[];weights=[];counts={}
    for target,threshold in thresholds.items():
        mask=up>=threshold if target else up<=threshold
        candidates=np.flatnonzero(mask)
        order=np.argsort(-up[candidates] if target else up[candidates],kind='stable')
        cap=int((labelled.label==target).sum()*settings['maximum_pseudo_to_labelled_ratio_per_class'])
        selected=candidates[order[:cap]]
        batch=unknown.iloc[selected].copy()
        batch['label']=target
        batches.append(batch)
        weights.extend([settings['sample_weight']]*len(batch))
        counts[target]=len(batch)
    combined=pd.concat([labelled,*batches],ignore_index=True)
    model=LGBMClassifier(**PARAMS)
    model.fit(combined[cols],combined.label.astype(int),sample_weight=np.r_[np.ones(len(labelled)),weights])
    p=model.predict_proba(future[cols])[:,list(model.classes_).index(1)]
    (out/f'{tag}_pseudo_audit.json').write_text(json.dumps({'inner_train_end':inner_end,'calibration_steps':settings['calibration_steps'],
        'tested_thresholds':audit,'selected_thresholds':thresholds,'pseudo_counts':counts,'labelled_rows':len(labelled),
        'warning':'Calibration precision is an empirical estimate on known labels; unknown-label precision is not observed.'},indent=2))
    return model,p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='new_signal_01')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in {'.','..'}:
        parser.error('Invalid run ID')
    out=HERE/'runs'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    cfg=HERE/'configs/new_signal.json';config=json.loads(cfg.read_text())
    paths=[Path(__file__),HERE/'pipeline.py',cfg]
    manifest={'config':config,'python':sys.version,
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','catboost','scikit-learn','scipy']},
              'sources':{p.name:sha256(p) for p in paths},
              'inputs':{f:sha256(HERE.parent/f) for f in ['train.csv','test.csv','txs_edgelist.csv','sample_submission.csv','submission_v2.csv']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for path in paths:
        (out/('source_'+path.name)).write_bytes(path.read_bytes())
    train,test,edges,sample=load_data(HERE.parent)
    train,test,gcols,_=graph_features(train,test,edges);cols=BASE+gcols
    records=[];steps=[];sensitivity=[]
    def fit(name,cutoff,va,tag):
        tr=train.loc[train.label.notna() & (train.time_step<=cutoff)]
        if name=='ordered_catboost':
            model=CatBoostClassifier(**config['catboost'])
            model.fit(tr[cols],tr.label.astype(int))
            p=model.predict_proba(va[cols])[:,list(model.classes_).index(1)]
        else:
            model,p=pseudo_fit(train,va,cols,cutoff,config['pseudo'],out,tag)
        joblib.dump(model,out/f'{tag}_model.joblib',compress=3)
        return p
    def combine(name,base,component):
        if name=='ordered_catboost':
            # Global percentiles align score scales; rank inputs use no label values.
            return (1-config['candidate_weight'])*percentile(base)+config['candidate_weight']*percentile(component)
        return (1-config['candidate_weight'])*base+config['candidate_weight']*component
    def evaluate(cutoff,names,phase):
        tr=train.loc[train.label.notna() & (train.time_step<=cutoff)]
        va=train.loc[train.label.notna() & train.time_step.between(cutoff+1,cutoff+14)]
        _,base=fit_predict(tr,va,cols,PARAMS)
        baseline_auc=roc_auc_score(va.label,base)
        predictions=va[['txId','time_step','label']].copy();predictions['v2']=base
        for name in names:
            p=fit(name,cutoff,va,f'{phase}_{cutoff}_{name}')
            blend=combine(name,base,p)
            predictions[name+'_raw']=p;predictions[name]=blend
            result={'phase':phase,'train_end':cutoff,'model':name,'baseline_auc':baseline_auc,'raw_auc':roc_auc_score(va.label,p),
                    'delta_auc':roc_auc_score(va.label,blend)-baseline_auc,**metrics(va.label,blend)}
            records.append(result)
            for step in sorted(va.time_step.unique()):
                mask=va.time_step.to_numpy()==step
                steps.append({'phase':phase,'train_end':cutoff,'time_step':int(step),'model':name,
                    'baseline_auc':roc_auc_score(va.label[mask],base[mask]) if va.label[mask].nunique()==2 else np.nan,
                    **metrics(va.label[mask],blend[mask])})
                sensitivity.append({'phase':phase,'train_end':cutoff,'excluded_step':int(step),'model':name,
                    'delta_auc':roc_auc_score(va.label[~mask],blend[~mask])-roc_auc_score(va.label[~mask],base[~mask])})
            pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
            pd.DataFrame(steps).to_csv(out/'metrics_by_step.csv',index=False)
            pd.DataFrame(sensitivity).to_csv(out/'leave_one_step_out.csv',index=False)
            predictions.to_csv(out/f'{phase}_{cutoff}_predictions.csv',index=False)
            print(json.dumps(result),flush=True)
    names=['ordered_catboost','pseudo_labels']
    for cutoff in config['development_origins']:
        evaluate(cutoff,names,'development')
    summary=pd.DataFrame(records).groupby('model').delta_auc.agg(['mean','min'])
    eligible=summary[(summary['mean']>0)&(summary['min']>=-.001)].sort_values('mean',ascending=False)
    selected=eligible.index[0] if len(eligible) else None
    decision={'development_summary':summary.to_dict(orient='index'),'frozen_candidate':selected}
    (out/'selection_before_confirmation.json').write_text(json.dumps(decision,indent=2))
    if selected:
        for cutoff in config['confirmation_origins']:
            evaluate(cutoff,[selected],'confirmation')
        deltas=pd.DataFrame(records).query("phase=='confirmation'").delta_auc
        advance=bool(deltas.mean()>0 and deltas.min()>=-.001)
    else:
        advance=False
    decision['passes_confirmation']=advance
    if advance:
        component=fit(selected,35,test,'final_'+selected)
        ref=pd.read_csv(HERE.parent/'submission_v2.csv')
        validate_submission(ref,test,sample)
        scores=combine(selected,ref.target.to_numpy(),component)
        # Rank scores are valid [0,1] submission scores, not calibrated probabilities.
        submission=pd.DataFrame({'index':test['index'],'target':scores})
        validate_submission(submission,test,sample)
        filename=f'submission_{selected}.csv'
        submission.to_csv(out/filename,index=False)
        pd.DataFrame({'index':test['index'],'component_probability':component}).to_csv(out/'final_component.csv',index=False)
        (out/'features.json').write_text(json.dumps(cols,indent=2))
        decision.update(submission_file=filename,submission_sha256=sha256(out/filename),rank_correlation_vs_v2=float(submission.target.corr(ref.target,method='spearman')))
    (out/'decision.json').write_text(json.dumps(decision,indent=2))
    (out/'COMPLETE').write_text('Completed successfully\n')
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    main()

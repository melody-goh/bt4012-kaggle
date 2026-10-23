"""Label-free graph risk propagation from historical-model predictions.

All neighbour scores come from the same model trained before the validation
period. No actual neighbour labels enter prediction or propagation. Isolated
nodes keep their base scores. Only supplied-feature nodes participate.
"""
import argparse
from pathlib import Path
import importlib.metadata
import json
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from pipeline import BASE,load_data,graph_features,fit_predict,metrics,validate_submission,sha256

HERE=Path(__file__).resolve().parent
PARAMS=dict(n_estimators=300,learning_rate=.03,num_leaves=31,random_state=42,verbosity=-1,n_jobs=4)


def neighbours(nodes,edges):
    lookup=pd.Series(np.arange(len(nodes)),index=nodes.txId)
    source=edges.txId1.map(lookup)
    target=edges.txId2.map(lookup)
    known=source.notna() & target.notna()
    incoming=csr_matrix((np.ones(int(known.sum())),(target[known].astype(int),source[known].astype(int))),shape=(len(nodes),len(nodes)))
    # Binary adjacency prevents reciprocal edges receiving double weight in undirected mode.
    both=incoming+incoming.T
    both.data[:]=1
    return {'upstream':incoming,'downstream':incoming.T.tocsr(),'undirected':both}


def propagate(scores,matrix,alpha,log_odds=False):
    if not 0<=alpha<=1:
        raise ValueError('alpha must lie in [0,1]')
    degree=np.asarray(matrix.sum(axis=1)).ravel()
    normalized=diags(1/np.maximum(degree,1))@matrix
    values=logit(np.clip(scores,1e-6,1-1e-6)) if log_odds else scores
    neighbor_values=normalized@values
    result=values.copy()
    linked=degree>0
    result[linked]=(1-alpha)*values[linked]+alpha*neighbor_values[linked]
    result=expit(result) if log_odds else result
    result[~linked]=scores[~linked]
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='graph_risk_propagation_01')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in {'.','..'}:
        parser.error('Invalid run ID')
    out=HERE/'runs'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    cfg=HERE/'configs/graph_risk_propagation.json'
    config=json.loads(cfg.read_text())
    manifest={'config':config,'base_parameters':PARAMS,'python':sys.version,
              'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','scikit-learn','scipy']},
              'inputs':{f:sha256(HERE.parent/f) for f in ['train.csv','test.csv','txs_edgelist.csv','sample_submission.csv','submission_v2.csv']},
              'sources':{p.name:sha256(p) for p in [Path(__file__),HERE/'pipeline.py',cfg]}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for p in [Path(__file__),HERE/'pipeline.py',cfg]:
        (out/('source_'+p.name)).write_bytes(p.read_bytes())
    train,test,edges,sample=load_data(HERE.parent)
    train,test,gcols,audit=graph_features(train,test,edges)
    cols=BASE+gcols
    records=[];steps=[];sensitivity=[]
    def evaluate(cutoff,names,phase):
        tr=train.loc[train.label.notna() & (train.time_step<=cutoff)]
        future=train.loc[train.time_step.between(cutoff+1,cutoff+14)].reset_index(drop=True)
        _,base=fit_predict(tr,future,cols,PARAMS)
        adjacency=neighbours(future,edges)
        labelled=future.label.notna().to_numpy()
        y=future.loc[labelled,'label']
        base_auc=roc_auc_score(y,base[labelled])
        predictions=future[['txId','time_step','label']].copy()
        predictions['v2']=base
        for name in names:
            matrix=adjacency[name.split('_')[0]]
            p=propagate(base,matrix,config['alpha'],name.endswith('logodds'))
            predictions[name]=p
            record={'phase':phase,'train_end':cutoff,'model':name,'baseline_auc':base_auc,
                    'delta_auc':roc_auc_score(y,p[labelled])-base_auc,**metrics(y,p[labelled])}
            records.append(record)
            for step in sorted(future.time_step.unique()):
                mask=labelled & (future.time_step.to_numpy()==step)
                step_y=future.loc[mask,'label']
                if not len(step_y):
                    continue
                steps.append({'phase':phase,'train_end':cutoff,'time_step':int(step),'model':name,
                              'baseline_auc':roc_auc_score(step_y,base[mask]) if step_y.nunique()==2 else np.nan,
                              **metrics(step_y,p[mask])})
                keep=labelled & ~mask
                sensitivity.append({'phase':phase,'train_end':cutoff,'excluded_step':int(step),'model':name,
                                    'delta_auc':roc_auc_score(future.loc[keep,'label'],p[keep])-roc_auc_score(future.loc[keep,'label'],base[keep])})
            print(json.dumps(record),flush=True)
        predictions.to_csv(out/f'{phase}_{cutoff}_predictions.csv',index=False)
        pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(steps).to_csv(out/'metrics_by_step.csv',index=False)
        pd.DataFrame(sensitivity).to_csv(out/'leave_one_step_out.csv',index=False)
    for cutoff in config['development_origins']:
        evaluate(cutoff,config['candidates'],'development')
    summary=pd.DataFrame(records).groupby('model').delta_auc.agg(['mean','min'])
    eligible=summary[(summary['mean']>0)&(summary['min']>=-.001)].sort_values('mean',ascending=False)
    chosen=eligible.index[0] if len(eligible) else None
    decision={'development_summary':summary.to_dict(orient='index'),'frozen_candidate':chosen}
    (out/'selection_before_confirmation.json').write_text(json.dumps(decision,indent=2))
    if chosen:
        for cutoff in config['confirmation_origins']:
            evaluate(cutoff,[chosen],'confirmation')
        deltas=pd.DataFrame(records).query("phase=='confirmation'").delta_auc
        approved=bool(deltas.mean()>0 and deltas.min()>=-.001)
    else:
        approved=False
    decision['passes_confirmation']=approved
    if approved:
        # Exact scored V2 scores, graph constructed without labels.
        reference=pd.read_csv(HERE.parent/'submission_v2.csv')
        validate_submission(reference,test,sample)
        adjacency=neighbours(test,edges)
        p=propagate(reference.target.to_numpy(),adjacency[chosen.split('_')[0]],config['alpha'],chosen.endswith('logodds'))
        submission=pd.DataFrame({'index':test['index'],'target':p})
        validate_submission(submission,test,sample)
        file=f'submission_{chosen}.csv'
        submission.to_csv(out/file,index=False)
        validate_submission(pd.read_csv(out/file),test,sample)
        decision.update(submission_file=file,submission_sha256=sha256(out/file),rank_correlation_vs_v2=float(submission.target.corr(reference.target,method='spearman')))
    (out/'decision.json').write_text(json.dumps(decision,indent=2))
    (out/'COMPLETE').write_text('Completed successfully\n')
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    main()

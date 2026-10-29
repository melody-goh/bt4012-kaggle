"""One predeclared graph-only candidate; no hyperparameter or blend-weight search."""
import json
from pathlib import Path
import sys
import importlib.metadata
import numpy as np
import pandas as pd
from pipeline import BASE,load_data,graph_features,fit_predict,metrics,sha256,validate_submission

HERE=Path(__file__).resolve().parent
out=HERE/'runs/graph_bagging_initial'
out.mkdir(parents=True,exist_ok=False)
config=json.loads((HERE/'configs/graph_bagging.json').read_text())
source=HERE/'runs/submission_diagnosis'
manifest={'config':config,'python':sys.version,
          'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm']},
          'source_hashes':{p.name:sha256(p) for p in [Path(__file__),HERE/'pipeline.py',HERE/'configs/graph_bagging.json']},
          'input_hashes':{f:sha256(HERE.parent/f) for f in ['train.csv','test.csv','txs_edgelist.csv','sample_submission.csv']}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
train,test,edges,sample=load_data(HERE.parent)
train,test,gcols,_=graph_features(train,test,edges)
results=[]
steps=[]
for cutoff in config['origins']:
    tr=train.loc[train.label.notna() & (train.time_step<=cutoff)]
    va=train.loc[train.label.notna() & train.time_step.between(cutoff+1,cutoff+14)]
    ref=pd.read_csv(source/f'predictions_cutoff_{cutoff}.csv')
    if not np.array_equal(ref.txId.to_numpy(),va.txId.to_numpy()):
        raise ValueError('Reference prediction alignment failed')
    pred=[]
    for seed in config['seeds']:
        model,p=fit_predict(tr,va,BASE+gcols,config['parameters']|{'random_state':seed})
        model.booster_.save_model(str(out/f'model_cutoff_{cutoff}_seed_{seed}.txt'))
        pred.append(p)
    candidate=config['v2_weight']*ref.v2.to_numpy()+(1-config['v2_weight'])*np.mean(pred,axis=0)
    frame=va[['txId','time_step','label']].copy()
    frame['v2']=ref.v2.to_numpy()
    frame['candidate']=candidate
    frame.to_csv(out/f'predictions_cutoff_{cutoff}.csv',index=False)
    for name in ['v2','candidate']:
        results.append({'train_end':cutoff,'model':name,**metrics(frame.label,frame[name])})
        for step,group in frame.groupby('time_step'):
            steps.append({'train_end':cutoff,'time_step':step,'model':name,**metrics(group.label,group[name])})
    print('Completed cutoff',cutoff,flush=True)
r=pd.DataFrame(results)
r.to_csv(out/'metrics.csv',index=False)
pd.DataFrame(steps).to_csv(out/'metrics_by_step.csv',index=False)
table=r.pivot(index='train_end',columns='model',values='roc_auc')
table['delta']=table.candidate-table.v2
advance=bool((table.loc[[14,21],'delta']>0).all() and table.loc[7,'delta']>=-.001)
(out/'decision.json').write_text(json.dumps({'advance':advance,'rule':config['advancement_rule'],'deltas':table.delta.to_dict()},indent=2))
if advance:
    tr=train.dropna(subset=['label'])
    final=[]
    for seed in config['seeds']:
        model,p=fit_predict(tr,test,BASE+gcols,config['parameters']|{'random_state':seed})
        model.booster_.save_model(str(out/f'final_seed_{seed}.txt'))
        final.append(p)
    reference=pd.read_csv(HERE.parent/'submission_v2.csv')
    validate_submission(reference,test,sample)
    submission=pd.DataFrame({'index':test['index'],'target':config['v2_weight']*reference.target.to_numpy()+(1-config['v2_weight'])*np.mean(final,axis=0)})
    validate_submission(submission,test,sample)
    submission.to_csv(out/'submission_graph_bagging.csv',index=False)
    validate_submission(pd.read_csv(out/'submission_graph_bagging.csv'),test,sample)
print(table.to_string(),flush=True)
print('Advance:',advance,flush=True)
(out/'COMPLETE').write_text('Completed successfully\n')

"""Reproducible diagnostic summaries and figures from saved predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs/submission_diagnosis'


def summarize():
    rows=[]
    intervals=[]
    rng=np.random.default_rng(42)
    for cutoff in [7,14,21]:
        d=pd.read_csv(OUT/f'predictions_cutoff_{cutoff}.csv')
        unique=np.sort(d.time_step.unique())
        codes=pd.Categorical(d.time_step,categories=unique).codes
        for step in unique:
            x=d[d.time_step!=step]
            rows.append({'train_end':cutoff,'excluded_step':step,'v3_minus_v2_auc':roc_auc_score(x.label,x.v3)-roc_auc_score(x.label,x.v2)})
        delta=[]
        # Paired resampling of whole time steps; descriptive historical uncertainty.
        # Does not model dependence between adjacent steps or future distribution shift.
        for _ in range(1000):
            counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
            weights=counts[codes]
            if weights[d.label==1].sum() and weights[d.label==0].sum():
                delta.append(roc_auc_score(d.label,d.v3,sample_weight=weights)-roc_auc_score(d.label,d.v2,sample_weight=weights))
        lo,hi=np.quantile(delta,[.025,.975])
        intervals.append({'train_end':cutoff,'v3_minus_v2_auc':roc_auc_score(d.label,d.v3)-roc_auc_score(d.label,d.v2),
                          'cluster_bootstrap_2.5pct':lo,'cluster_bootstrap_97.5pct':hi,'replicates':len(delta)})
    pd.DataFrame(rows).to_csv(OUT/'leave_one_step_out_diagnostic.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'historical_bootstrap.csv',index=False)
    steps=pd.read_csv(OUT/'metrics_by_step.csv')
    pivot=steps.pivot(index=['train_end','time_step'],columns='model',values='roc_auc')
    pivot['v3_minus_v2']=pivot.v3-pivot.v2
    pivot.to_csv(OUT/'step_auc_changes.csv')
    summaries=[]
    for cutoff,frame in pivot.groupby(level='train_end'):
        diff=frame.v3_minus_v2
        summaries.append({'train_end':cutoff,'steps_improved':int((diff>1e-12).sum()),
                          'steps_worsened':int((diff< -1e-12).sum()),'steps_tied':int((diff.abs()<=1e-12).sum()),
                          'macro_v2_auc':frame.v2.mean(),'macro_v3_auc':frame.v3.mean()})
    pd.DataFrame(summaries).to_csv(OUT/'step_win_loss_summary.csv',index=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors={'v1':'#3775a9','v2':'#24876a','v3':'#b64d44'}
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True)
    for ax,cutoff in zip(axes,[7,14,21]):
        frame=pivot.loc[cutoff]
        ax.axhline(0,color='#555',linewidth=.8)
        ax.bar(frame.index,frame.v3_minus_v2,color=[colors['v2'] if v>=0 else colors['v3'] for v in frame.v3_minus_v2])
        ax.set(title=f'Train 1–{cutoff}; evaluate {cutoff+1}–{cutoff+14}',xlabel='Validation time step')
        ax.set_xticks(frame.index[::2])
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('V3 ROC-AUC minus V2 ROC-AUC')
    fig.suptitle('V3 gains are concentrated in a few historical periods',fontsize=14)
    fig.tight_layout(rect=(0,.05,1,.93))
    fig.text(.02,.015,'Positive = V3 better. Legacy windows overlap; they are not independent tests. Source: reconstructed models, supplied training labels.',fontsize=9)
    fig.savefig(OUT/'per_step_auc_changes.png',dpi=170)
    plt.close(fig)
    metrics=pd.read_csv(OUT/'metrics.csv').query("model in ['v1','v2','v3']")
    scores=pd.read_csv(OUT/'leaderboard.csv').set_index('model').public_roc_auc
    fig,axes=plt.subplots(1,4,figsize=(12,3.8))
    for ax,cutoff in zip(axes[:3],[7,14,21]):
        frame=metrics[metrics.train_end==cutoff].set_index('model').roc_auc
        for i,name in enumerate(['v1','v2','v3']):
            ax.scatter(i,frame[name],color=colors[name],s=55)
            ax.annotate(f'{frame[name]:.5f}',(i,frame[name]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.set_title(f'Historical {cutoff+1}–{cutoff+14}')
    for i,name in enumerate(['v1','v2','v3']):
        axes[3].scatter(i,scores[name],color=colors[name],s=55)
        axes[3].annotate(f'{scores[name]:.5f}',(i,scores[name]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    axes[3].set_title('Kaggle public (reported)')
    for ax in axes:
        ax.set_xticks([0,1,2],['V1','V2','V3'])
        ax.set_xlim(-.6,2.6)
        ax.margins(y=.4)
        ax.grid(axis='y',alpha=.2)
        ax.ticklabel_format(useOffset=False,axis='y')
    axes[0].set_ylabel('ROC-AUC (each panel uses its own scale)')
    fig.suptitle('Historical ranking does not consistently match the public leaderboard',fontsize=13)
    fig.tight_layout(rect=(0,.08,1,.91))
    fig.text(.02,.02,'Exact values shown; y-axis scales differ. Public scores from the Kaggle leaderboard. No hidden-test labels were accessed.',fontsize=9)
    fig.savefig(OUT/'local_vs_public_auc.png',dpi=170)
    plt.close(fig)
    print(pd.DataFrame(intervals).round(6).to_string(index=False))
    print(pd.DataFrame(summaries).round(6).to_string(index=False))


if __name__=='__main__':
    summarize()

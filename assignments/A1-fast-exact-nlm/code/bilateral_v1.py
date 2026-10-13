"""Structural bilateral filtering, frozen three-route V1, and Optuna tuning."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from skimage.metrics import structural_similarity
from skimage.transform import resize

from bilateral_dataset_pipeline import POPCOUNT, lbp8, read_image, tau_from
from bilateral_foundations import (BOUNDARY, append_log, bilateral_from_scratch,
                                   dataframe_markdown, gaussian_from_scratch,
                                   metrics, recommended_k, spatial_kernel,
                                   synthetic_cases, validate_window)


CODE_VERSION="v1-routing-1.0"
TUNE_SEED=20260827
VALIDATION_SEED=20260927
METHODS=("SBF_always_on","G_BF_gate1","BF_SBF_conflict_gate","G_BF_SBF_full_V1")


def _pad_mode(boundary: str) -> str:
    if boundary=="reflect": return "symmetric"
    if boundary=="mirror": return "reflect"
    raise ValueError("boundary must be reflect or mirror")


def structural_filter(image: np.ndarray, method: str, sigma_s: float, sigma_r: float,
                      epsilon: float, sigma_t: float, h: float, T_A: float,
                      k: int | None=None, boundary: str=BOUNDARY,
                      chunk_rows: int=32) -> tuple[np.ndarray,dict,dict[str,np.ndarray]]:
    """Evaluate an always-on or selectively routed structural bilateral variant.

    Route map values: 0=G, 1=BF, 2=SBF. The frozen full V1 checks H<=tau
    before cue conflict, exactly as specified in the source document.
    """
    if method not in METHODS: raise ValueError(f"method must be one of {METHODS}")
    if sigma_r<=0 or sigma_t<=0 or not (0<epsilon<1) or not (0<=h<=1) or not (0<=T_A<=1):
        raise ValueError("invalid range/structural/gating parameters")
    k=recommended_k(sigma_s) if k is None else k; radius=validate_window(k); qcount=k*k
    x=np.asarray(image,dtype=np.float64); code=lbp8(x); gs=spatial_kernel(k,sigma_s); gs_sum=float(gs.sum())
    pad=_pad_mode(boundary); px=np.pad(x,radius,mode=pad); pc=np.pad(code,radius,mode=pad)
    windows=sliding_window_view(px,(k,k)); code_windows=sliding_window_view(pc,(k,k))
    out=np.empty_like(x); neff=np.empty_like(x); route=np.empty(x.shape,dtype=np.uint8)
    conflict=np.zeros_like(x,dtype=np.float32); local_H=np.empty_like(x,dtype=np.float32)
    tau=tau_from(sigma_r,epsilon); inv_r=1/(2*sigma_r*sigma_r); inv_t=1/(2*sigma_t*sigma_t)
    range_evals=struct_evals=range_exps=struct_exps=comparisons=0
    start=time.perf_counter()
    for y0 in range(0,x.shape[0],chunk_rows):
        y1=min(y0+chunk_rows,x.shape[0]); win=windows[y0:y1]; cwin=code_windows[y0:y1]
        centers=x[y0:y1]; center_codes=code[y0:y1]
        H=win.max(axis=(-2,-1))-win.min(axis=(-2,-1)); local_H[y0:y1]=H
        gout=np.einsum("ijkl,kl->ij",win,gs,optimize=True)/gs_sum
        g_neff=gs_sum*gs_sum/float((gs*gs).sum())
        out_chunk=gout.copy(); neff_chunk=np.full(centers.shape,g_neff); route_chunk=np.zeros(centers.shape,dtype=np.uint8)
        if method=="SBF_always_on": active=np.ones(centers.shape,dtype=bool)
        elif method=="BF_SBF_conflict_gate": active=np.ones(centers.shape,dtype=bool)
        else: active=H>tau; comparisons += centers.size
        if np.any(active):
            awin=win[active]; delta=centers[active,None,None]-awin
            if method=="G_BF_gate1":
                gr=np.exp(-(delta*delta)*inv_r); weights=gr*gs; den=weights.sum((-2,-1))
                out_chunk[active]=(weights*awin).sum((-2,-1))/den
                neff_chunk[active]=den*den/np.maximum((weights*weights).sum((-2,-1)),1e-15); route_chunk[active]=1
                nactive=int(active.sum()); range_evals+=nactive*qcount; range_exps+=nactive*qcount
            else:
                acwin=cwin[active]; acode=center_codes[active,None,None]
                dt=POPCOUNT[np.bitwise_xor(acode,acwin)]/8.0
                nactive=int(active.sum()); struct_evals+=nactive*qcount
                if method=="SBF_always_on":
                    sbf_mask=np.ones(nactive,dtype=bool); A=np.zeros(nactive,dtype=np.float32)
                else:
                    A=((np.abs(delta)<=tau)&(dt>h)).mean(axis=(-2,-1)).astype(np.float32)
                    conflict[y0:y1][active]=A; comparisons += 2*nactive*qcount+nactive
                    sbf_mask=A>T_A
                gr=np.exp(-(delta*delta)*inv_r); range_evals+=nactive*qcount; range_exps+=nactive*qcount
                weights_bf=gr*gs; den_bf=weights_bf.sum((-2,-1)); bf_vals=(weights_bf*awin).sum((-2,-1))/den_bf
                bf_neff=den_bf*den_bf/np.maximum((weights_bf*weights_bf).sum((-2,-1)),1e-15)
                vals=bf_vals.copy(); nvals=bf_neff.copy(); local_routes=np.ones(nactive,dtype=np.uint8)
                if np.any(sbf_mask):
                    gt=np.exp(-(dt[sbf_mask]*dt[sbf_mask])*inv_t); ws=weights_bf[sbf_mask]*gt
                    ds=ws.sum((-2,-1)); vals[sbf_mask]=(ws*awin[sbf_mask]).sum((-2,-1))/ds
                    nvals[sbf_mask]=ds*ds/np.maximum((ws*ws).sum((-2,-1)),1e-15); local_routes[sbf_mask]=2
                    struct_exps+=int(sbf_mask.sum())*qcount
                out_chunk[active]=vals; neff_chunk[active]=nvals; route_chunk[active]=local_routes
        out[y0:y1]=out_chunk; neff[y0:y1]=neff_chunk; route[y0:y1]=route_chunk
    runtime_ms=(time.perf_counter()-start)*1000
    n=x.size; fractions=[float(np.mean(route==r)) for r in range(3)]
    diag=dict(method=method,K=k,radius=radius,sigma_s=sigma_s,sigma_r=sigma_r,epsilon=epsilon,tau=tau,
              sigma_t=sigma_t,h=h,T_A=T_A,runtime_ms=runtime_ms,range_eval_count=range_evals,
              structural_eval_count=struct_evals,exponential_count=range_exps+struct_exps,
              range_exponential_count=range_exps,structural_exponential_count=struct_exps,
              comparison_count=comparisons,P_G=fractions[0],P_BF=fractions[1],P_SBF=fractions[2],
              mean_Neff=float(neff.mean()),multiplication_count=np.nan,addition_count=np.nan,division_count=np.nan)
    maps=dict(route=route,conflict=conflict,local_range=local_H,neff=neff.astype(np.float32))
    return out.astype(np.float32),diag,maps


def synthetic_gate_proofs(out_dir: Path, params: dict) -> pd.DataFrame:
    out_dir.mkdir(parents=True,exist_ok=True); cases=synthetic_cases(); rows=[]
    for name in ("low_amplitude_periodic","intensity_similar_structural_conflict","step_edge"):
        clean=cases[name]
        outputs={"input":clean}
        for method in ("G_BF_gate1","SBF_always_on","G_BF_SBF_full_V1"):
            out,diag,maps=structural_filter(clean,method=method,**params); outputs[method]=out
            amplitude=float(np.percentile(out,95)-np.percentile(out,5))
            rows.append(dict(case=name,method=method,output_p05_p95_amplitude=amplitude,
                             input_p05_p95_amplitude=float(np.percentile(clean,95)-np.percentile(clean,5)),
                             **{k:diag[k] for k in ["tau","P_G","P_BF","P_SBF","mean_Neff"]}))
        fig,axes=plt.subplots(1,4,figsize=(13,3.3))
        for ax,(label,data) in zip(axes,outputs.items()): ax.imshow(data,cmap="gray",vmin=0,vmax=255); ax.set_title(label,fontsize=8); ax.axis("off")
        fig.suptitle(name); fig.tight_layout(); fig.savefig(out_dir/f"{name}_gate_proof.png",dpi=160); plt.close(fig)
    df=pd.DataFrame(rows); df.to_csv(out_dir/"gate_proof_metrics.csv",index=False)
    periodic=df[df.case.eq("low_amplitude_periodic")]
    full=periodic[periodic.method.eq("G_BF_SBF_full_V1")].iloc[0]
    conclusion=("The frozen full V1 still routes the low-amplitude periodic case entirely to Gaussian; "
                "the conflict cue cannot veto Gate 1 because H<=tau exits first. This is a documented negative result and motivates a later texture-risk-veto ablation."
                if full.P_G>.999 else "The chosen tau is small enough that the periodic case reaches the conflict stage.")
    (out_dir/"gate_proof_report.md").write_text("# Synthetic gate proofs\n\n"+dataframe_markdown(df,5)+"\n\n## Finding\n\n"+conclusion+"\n")
    return df


def load_selected(selection_csv: Path, dataset_dir: Path, side: int) -> dict[str,dict]:
    selected=pd.read_csv(selection_csv); data={}
    for _,r in selected.iterrows():
        _,native,_=read_image(dataset_dir/r.relative_path)
        clean=resize(native,(side,side),order=1,mode="reflect",anti_aliasing=True,preserve_range=True).astype(np.float32) if native.shape!=(side,side) else native.astype(np.float32)
        data[r.image_id]=dict(clean=clean,role=r.role,filename=r.filename,relative_path=r.relative_path,
                              native_resolution=f"{int(r.height)}x{int(r.width)}")
    return data


def noisy_bank(data: dict[str,dict], noise_levels: tuple[int,...], seed: int) -> dict[tuple[str,int],np.ndarray]:
    bank={}
    for idx,(image_id,item) in enumerate(data.items()):
        for sigma_n in noise_levels:
            rng=np.random.default_rng(seed+idx*1000+sigma_n)
            bank[(image_id,sigma_n)]=np.clip(item["clean"]+rng.normal(0,sigma_n,item["clean"].shape),0,255).astype(np.float32)
    return bank


def suggest_params(trial: optuna.Trial) -> dict:
    sigma_s=trial.suggest_float("sigma_s",.7,1.8,step=.1)
    return dict(sigma_s=sigma_s,sigma_r=trial.suggest_float("sigma_r",20,100,step=5),
                epsilon=trial.suggest_categorical("epsilon",[.005,.01,.025,.05,.10]),
                sigma_t=trial.suggest_float("sigma_t",.10,.60,step=.05),
                h=trial.suggest_categorical("h",[.125,.25,.375,.50,.625,.75]),
                T_A=trial.suggest_float("T_A",.02,.40,step=.02),k=recommended_k(sigma_s),boundary=BOUNDARY)


def tune_optuna(selection_csv: Path, dataset_dir: Path, out_dir: Path,
                n_global_trials: int=36, n_per_image_trials: int=12) -> dict:
    out_dir.mkdir(parents=True,exist_ok=True); optuna.logging.set_verbosity(optuna.logging.WARNING)
    data=load_selected(selection_csv,dataset_dir,128); bank=noisy_bank(data,(10,25,50),TUNE_SEED)
    def make_objective(ids: list[str]):
        def objective(trial: optuna.Trial) -> float:
            p=suggest_params(trial); ssims=[]; exp_ratios=[]; struct_ratios=[]
            for image_id in ids:
                clean=data[image_id]["clean"]
                for sigma_n in (10,25,50):
                    out,diag,_=structural_filter(bank[(image_id,sigma_n)],"G_BF_SBF_full_V1",**p)
                    ssims.append(structural_similarity(clean,out,data_range=255))
                    denom=clean.size*p["k"]*p["k"]
                    exp_ratios.append(diag["exponential_count"]/denom); struct_ratios.append(diag["structural_eval_count"]/denom)
            mean_ssim=float(np.mean(ssims)); exp_ratio=float(np.mean(exp_ratios)); struct_ratio=float(np.mean(struct_ratios))
            score=mean_ssim-.015*exp_ratio-.005*struct_ratio
            trial.set_user_attr("mean_ssim",mean_ssim); trial.set_user_attr("mean_exp_ratio",exp_ratio); trial.set_user_attr("mean_struct_ratio",struct_ratio)
            return score
        return objective
    sampler=optuna.samplers.TPESampler(seed=TUNE_SEED,multivariate=True)
    global_study=optuna.create_study(direction="maximize",sampler=sampler,study_name="global_v1")
    global_study.optimize(make_objective(list(data)),n_trials=n_global_trials,show_progress_bar=False)
    gdf=global_study.trials_dataframe(); gdf.to_csv(out_dir/"optuna_global_trials.csv",index=False)
    best_global=dict(global_study.best_params); best_global["k"]=recommended_k(best_global["sigma_s"]); best_global["boundary"]=BOUNDARY
    per_image={}; per_rows=[]
    for idx,image_id in enumerate(data):
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=TUNE_SEED+idx+1),study_name=f"diag_{image_id}")
        study.optimize(make_objective([image_id]),n_trials=n_per_image_trials,show_progress_bar=False)
        p=dict(study.best_params); p["k"]=recommended_k(p["sigma_s"]); p["boundary"]=BOUNDARY; per_image[image_id]=p
        row=dict(image_id=image_id,role=data[image_id]["role"],best_score=study.best_value,**p); per_rows.append(row)
        study.trials_dataframe().to_csv(out_dir/f"optuna_{image_id}_diagnostic_trials.csv",index=False)
    payload=dict(protocol=dict(tune_side=128,noise_levels=[10,25,50],seed=TUNE_SEED,
                               objective="mean_SSIM - 0.015*exponential_ratio - 0.005*structural_eval_ratio",
                               global_trials=n_global_trials,per_image_trials=n_per_image_trials),
                 global_best=best_global,global_best_score=global_study.best_value,per_image_diagnostic=per_image)
    (out_dir/"best_parameters.json").write_text(json.dumps(payload,indent=2))
    pd.DataFrame(per_rows).to_csv(out_dir/"per_image_diagnostic_best.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(gdf.number,gdf.value,alpha=.4,label="trial score")
    ax.plot(gdf.number,gdf.value.cummax(),lw=2,label="best so far"); ax.set(title="Optuna global V1 convergence",xlabel="trial",ylabel="objective"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out_dir/"optuna_convergence.png",dpi=160); plt.close(fig)
    return payload


def validate_and_ablate(selection_csv: Path,dataset_dir: Path,out_dir: Path,experiment_log: Path,
                        params_payload: dict) -> pd.DataFrame:
    out_dir.mkdir(parents=True,exist_ok=True); data=load_selected(selection_csv,dataset_dir,256)
    bank=noisy_bank(data,(10,25,50),VALIDATION_SEED); p=dict(params_payload["global_best"])
    run_id=datetime.now(timezone.utc).strftime("v1_%Y%m%dT%H%M%SZ"); code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows=[]
    for image_id,item in data.items():
        clean=item["clean"]
        for sigma_n in (10,25,50):
            noisy=bank[(image_id,sigma_n)]
            bf,bfdiag,bfneff=bilateral_from_scratch(noisy,p["sigma_s"],p["sigma_r"],p["k"],BOUNDARY,return_neff=True)
            g,gdiag=gaussian_from_scratch(noisy,p["sigma_s"],p["k"],BOUNDARY)
            gs=spatial_kernel(p["k"],p["sigma_s"]); gneff=float(gs.sum()**2/(gs*gs).sum())
            variants=[("G",g,{**gdiag,"epsilon":np.nan,"tau":np.nan,"sigma_t":np.nan,"h":np.nan,"T_A":np.nan,
                              "P_G":1.,"P_BF":0.,"P_SBF":0.,"mean_Neff":gneff},None),
                      ("BF",bf,{**bfdiag,"epsilon":np.nan,"tau":np.nan,"sigma_t":np.nan,"h":np.nan,"T_A":np.nan,
                                "P_G":0.,"P_BF":1.,"P_SBF":0.,"mean_Neff":float(bfneff.mean())},None)]
            for method in METHODS:
                out,diag,maps=structural_filter(noisy,method=method,**p); variants.append((method,out,diag,maps))
            for method,out,diag,maps in variants:
                met=metrics(clean,out); record=dict(timestamp=datetime.now(timezone.utc).isoformat(),run_id=run_id,
                    code_version=CODE_VERSION,code_sha256=code_hash,image_id=image_id,source_path=item["relative_path"],
                    source_resolution=item["native_resolution"],experiment_resolution="256x256",grayscale_conversion="BT.601 or original L",
                    noise_type="additive_gaussian_clipped",noise_sigma=sigma_n,random_seed=VALIDATION_SEED,method=method,
                    K=diag["K"],radius=diag["radius"],sigma_s=diag["sigma_s"],sigma_r=diag.get("sigma_r",np.nan),
                    epsilon=diag.get("epsilon",np.nan),tau=diag.get("tau",np.nan),structural_descriptor="LBP P=8 R=1" if "SBF" in method else "none",
                    structural_parameters=f"sigma_t={diag.get('sigma_t',np.nan)}",h=diag.get("h",np.nan),T_A=diag.get("T_A",np.nan),
                    MSB_depth_b=np.nan,boundary=BOUNDARY,runtime_ms=diag["runtime_ms"],range_eval_count=diag["range_eval_count"],
                    structural_eval_count=diag["structural_eval_count"],exponential_count=diag["exponential_count"],
                    comparison_count=diag["comparison_count"],multiplication_count=diag.get("multiplication_count",np.nan),
                    addition_count=diag.get("addition_count",np.nan),division_count=diag.get("division_count",np.nan),
                    P_G=diag["P_G"],P_BF=diag["P_BF"],P_SBF=diag["P_SBF"],BF_approximation_MAE=float(np.mean(np.abs(out-bf))),
                    mean_Neff=diag["mean_Neff"],notes="Held-out seed; global Optuna parameters tuned at 128",**met)
                rows.append(record)
            # Main route/cue map for sigma_n=25.
            if sigma_n==25:
                full=[v for v in variants if v[0]=="G_BF_SBF_full_V1"][0]; maps=full[3]
                fig,axes=plt.subplots(1,4,figsize=(14,3.5))
                axes[0].imshow(clean,cmap="gray",vmin=0,vmax=255); axes[0].set_title("clean")
                axes[1].imshow(full[1],cmap="gray",vmin=0,vmax=255); axes[1].set_title("full V1")
                axes[2].imshow(maps["conflict"],cmap="inferno",vmin=0,vmax=1); axes[2].set_title("conflict A(p)")
                axes[3].imshow(maps["route"],cmap="viridis",vmin=0,vmax=2); axes[3].set_title("route 0=G,1=BF,2=SBF")
                for ax in axes: ax.axis("off")
                fig.suptitle(f"{image_id}: held-out V1 diagnostics"); fig.tight_layout(); fig.savefig(out_dir/f"{image_id}_v1_maps.png",dpi=160); plt.close(fig)
    append_log(rows,experiment_log); df=pd.DataFrame(rows); df.to_csv(out_dir/"v1_ablation_results.csv",index=False)
    agg=df.groupby("method").agg(PSNR_mean=("PSNR","mean"),SSIM_mean=("SSIM","mean"),edge_mean=("edge_metric","mean"),
                                    runtime_ms_mean=("runtime_ms","mean"),exp_count_mean=("exponential_count","mean"),
                                    P_G_mean=("P_G","mean"),P_BF_mean=("P_BF","mean"),P_SBF_mean=("P_SBF","mean"),
                                    BF_MAE_mean=("BF_approximation_MAE","mean")).reset_index().sort_values("SSIM_mean",ascending=False)
    agg.to_csv(out_dir/"v1_ablation_aggregate.csv",index=False)
    (out_dir/"v1_ablation_report.md").write_text("# Held-out V1 ablation\n\nGlobal parameters were tuned at 128 on seed "
        f"{TUNE_SEED} and evaluated here at 256 on unseen seed {VALIDATION_SEED}.\n\n## Global parameters\n\n```json\n"+
        json.dumps(p,indent=2)+"\n```\n\n## Aggregate results\n\n"+dataframe_markdown(agg,5)+
        "\n\nPer-image diagnostic Optuna values are not used for these final comparisons.\n")
    return df


def run_all(output_root: str|Path="outputs",n_global_trials: int=36,n_per_image_trials: int=12) -> dict:
    root=Path(output_root); tune_dir=root/"07_core_ablations"/"optuna"; v1_dir=root/"06_v1_filters"
    payload=tune_optuna(root/"02_image_selection/final_four.csv",Path("misc"),tune_dir,n_global_trials,n_per_image_trials)
    proofs=synthetic_gate_proofs(root/"05_synthetic_proofs",payload["global_best"])
    results=validate_and_ablate(root/"02_image_selection/final_four.csv",Path("misc"),root/"07_core_ablations",
                                root/"reports/experiment_log.csv",payload)
    return dict(parameters=payload,proofs=proofs,results=results)


if __name__=="__main__":
    result=run_all()
    print(json.dumps(result["parameters"]["global_best"],indent=2))
    print(pd.read_csv("outputs/07_core_ablations/v1_ablation_aggregate.csv").to_string(index=False))

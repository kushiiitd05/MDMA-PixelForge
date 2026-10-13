"""Recovery experiments after the frozen V1 routing failure.

Separately tunes exact BF, scikit-image BF, Gaussian-guided joint BF, and a
noise-corrected variance gate for each noise level. Tuning uses 128px/one seed;
evaluation uses 256px/a different seed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from skimage.metrics import structural_similarity
from skimage.restoration import denoise_bilateral

from bilateral_foundations import (append_log, bilateral_from_scratch, dataframe_markdown,
                                   gaussian_from_scratch, metrics, recommended_k,
                                   spatial_kernel, validate_window)
from bilateral_v1 import (BOUNDARY, TUNE_SEED, VALIDATION_SEED, load_selected,
                          noisy_bank)


NOISE_LEVELS=(10,25,50)
METHODS=("BF_exact","BF_skimage","JBF_gaussian_guide")


def joint_bilateral(image: np.ndarray, guide: np.ndarray, sigma_s: float,
                    sigma_r: float, k: int, chunk_rows: int=32) -> tuple[np.ndarray,float]:
    """Exact joint BF: weights from guide, weighted values from image."""
    radius=validate_window(k); x=np.asarray(image,dtype=np.float64); z=np.asarray(guide,dtype=np.float64)
    if x.shape!=z.shape: raise ValueError("image and guide must have the same shape")
    gs=spatial_kernel(k,sigma_s); px=np.pad(x,radius,mode="symmetric"); pz=np.pad(z,radius,mode="symmetric")
    xw=sliding_window_view(px,(k,k)); zw=sliding_window_view(pz,(k,k)); out=np.empty_like(x)
    inv=1/(2*sigma_r*sigma_r); start=time.perf_counter()
    for y0 in range(0,x.shape[0],chunk_rows):
        y1=min(y0+chunk_rows,x.shape[0]); wx=xw[y0:y1]; wz=zw[y0:y1]
        delta=z[y0:y1,:,None,None]-wz; weights=gs*np.exp(-(delta*delta)*inv); den=weights.sum((-2,-1))
        out[y0:y1]=(weights*wx).sum((-2,-1))/den
    return out.astype(np.float32),(time.perf_counter()-start)*1000


def skimage_bilateral(image: np.ndarray,sigma_s: float,sigma_r: float,k: int) -> tuple[np.ndarray,float]:
    start=time.perf_counter()
    out=denoise_bilateral(image.astype(np.float64)/255.0,win_size=k,sigma_color=sigma_r/255.0,
                          sigma_spatial=sigma_s,bins=10000,mode="symmetric",channel_axis=None)*255.0
    return out.astype(np.float32),(time.perf_counter()-start)*1000


def noise_corrected_signal_std(noisy: np.ndarray,sigma_n: float,k: int) -> np.ndarray:
    """Estimate local signal standard deviation from Var[Y] ~= Var[X]+sigma_n^2."""
    x=noisy.astype(np.float64); mean=ndimage.uniform_filter(x,size=k,mode="reflect")
    var=np.maximum(ndimage.uniform_filter(x*x,size=k,mode="reflect")-mean*mean,0)
    return np.sqrt(np.maximum(var-sigma_n*sigma_n,0)).astype(np.float32)


def apply_method(method: str,noisy: np.ndarray,p: dict) -> tuple[np.ndarray,float]:
    if method=="BF_exact":
        out,d=bilateral_from_scratch(noisy,p["sigma_s"],p["sigma_r"],p["k"],BOUNDARY); return out,d["runtime_ms"]
    if method=="BF_skimage": return skimage_bilateral(noisy,p["sigma_s"],p["sigma_r"],p["k"])
    if method=="JBF_gaussian_guide":
        guide=ndimage.gaussian_filter(noisy.astype(float),sigma=p["guide_sigma"],mode="reflect")
        return joint_bilateral(noisy,guide,p["sigma_s"],p["sigma_r"],p["k"])
    raise ValueError(method)


def tune_baselines(selection_csv: Path,dataset_dir: Path,out_dir: Path,n_trials: int=24) -> dict:
    out_dir.mkdir(parents=True,exist_ok=True); optuna.logging.set_verbosity(optuna.logging.WARNING)
    data=load_selected(selection_csv,dataset_dir,128); bank=noisy_bank(data,NOISE_LEVELS,TUNE_SEED)
    best={}; trial_frames=[]
    for method in METHODS:
        best[method]={}
        for ni,sigma_n in enumerate(NOISE_LEVELS):
            def objective(trial):
                sigma_s=trial.suggest_float("sigma_s",.7,2.1,step=.2)
                ratio=trial.suggest_float("sigma_r_over_sigma_n",.5,4.0,step=.25)
                p=dict(sigma_s=sigma_s,sigma_r=ratio*sigma_n,k=recommended_k(sigma_s))
                if method=="JBF_gaussian_guide": p["guide_sigma"]=trial.suggest_float("guide_sigma",.3,1.5,step=.2)
                vals=[]
                for image_id,item in data.items():
                    out,_=apply_method(method,bank[(image_id,sigma_n)],p)
                    vals.append(structural_similarity(item["clean"],out,data_range=255))
                return float(np.mean(vals))
            study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=TUNE_SEED+ni+100*METHODS.index(method)))
            study.optimize(objective,n_trials=n_trials,show_progress_bar=False)
            p=dict(study.best_params); p["sigma_r"]=p.pop("sigma_r_over_sigma_n")*sigma_n; p["k"]=recommended_k(p["sigma_s"])
            best[method][str(sigma_n)]=p
            tdf=study.trials_dataframe(); tdf.insert(0,"method",method); tdf.insert(1,"noise_sigma",sigma_n); trial_frames.append(tdf)
    pd.concat(trial_frames,ignore_index=True).to_csv(out_dir/"baseline_tuning_trials.csv",index=False)
    return dict(protocol=dict(side=128,seed=TUNE_SEED,noise_levels=list(NOISE_LEVELS),trials_per_method_noise=n_trials,
                              objective="mean SSIM across four image roles"),best=best)


def tune_noise_aware_gate(selection_csv: Path,dataset_dir: Path,best: dict,n_trials: int=24) -> dict:
    data=load_selected(selection_csv,dataset_dir,128); bank=noisy_bank(data,NOISE_LEVELS,TUNE_SEED); gate_best={}; edge_best={}
    for ni,sigma_n in enumerate(NOISE_LEVELS):
        bp=best["BF_exact"][str(sigma_n)]; cache={}
        for image_id,item in data.items():
            noisy=bank[(image_id,sigma_n)]; bf,_=apply_method("BF_exact",noisy,bp)
            g,_=gaussian_from_scratch(noisy,bp["sigma_s"],bp["k"],BOUNDARY)
            sig=noise_corrected_signal_std(noisy,sigma_n,bp["k"]); cache[image_id]=(bf,g,sig)
        def objective(trial):
            threshold=trial.suggest_float("signal_std_threshold",0,30,step=1)
            ssims=[]; pgs=[]
            for image_id,item in data.items():
                bf,g,sig=cache[image_id]; mask=sig<=threshold; out=np.where(mask,g,bf)
                ssims.append(structural_similarity(item["clean"],out,data_range=255)); pgs.append(mask.mean())
            # Quality first, with a very small preference for avoiding range work.
            return float(np.mean(ssims)+.002*np.mean(pgs))
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=TUNE_SEED+500+ni))
        study.optimize(objective,n_trials=n_trials,show_progress_bar=False)
        gate_best[str(sigma_n)]=dict(signal_std_threshold=study.best_params["signal_std_threshold"],score=study.best_value)
        guide_cache={}
        def edge_objective(trial):
            threshold=trial.suggest_float("signal_std_threshold",0,30,step=1)
            guide_sigma=trial.suggest_float("edge_guide_sigma",.7,1.5,step=.2)
            grad_threshold=trial.suggest_float("gradient_threshold",2,30,step=2)
            ssims=[]; pgs=[]; deltas=[]
            for image_id,item in data.items():
                bf,g,sig=cache[image_id]; key=(image_id,guide_sigma)
                if key not in guide_cache:
                    guide=ndimage.gaussian_filter(bank[(image_id,sigma_n)].astype(float),sigma=guide_sigma,mode="reflect")
                    guide_cache[key]=np.hypot(ndimage.sobel(guide,0,mode="reflect")/8,
                                              ndimage.sobel(guide,1,mode="reflect")/8)
                mask=(sig<=threshold)&(guide_cache[key]<=grad_threshold); out=np.where(mask,g,bf)
                score=structural_similarity(item["clean"],out,data_range=255)
                bf_score=structural_similarity(item["clean"],bf,data_range=255)
                ssims.append(score); deltas.append(score-bf_score); pgs.append(mask.mean())
            # Prevent a mean improvement from concealing serious damage to one role.
            return float(np.mean(ssims)+.002*np.mean(pgs)+.5*min(0.0,min(deltas)))
        edge_study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=TUNE_SEED+700+ni))
        edge_study.optimize(edge_objective,n_trials=max(n_trials,32),show_progress_bar=False)
        edge_best[str(sigma_n)]={**edge_study.best_params,"score":edge_study.best_value}
    return gate_best,edge_best


def tune_joint_edge_hybrid(selection_csv: Path,dataset_dir: Path,best: dict,n_trials: int=32) -> dict:
    data=load_selected(selection_csv,dataset_dir,128); bank=noisy_bank(data,NOISE_LEVELS,TUNE_SEED); result={}
    for ni,sigma_n in enumerate(NOISE_LEVELS):
        ep=best["BF_exact"][str(sigma_n)]; jp=best["JBF_gaussian_guide"][str(sigma_n)]; cache={}; grad_cache={}
        for image_id,item in data.items():
            noisy=bank[(image_id,sigma_n)]; exact,_=apply_method("BF_exact",noisy,ep); joint,_=apply_method("JBF_gaussian_guide",noisy,jp)
            cache[image_id]=(exact,joint)
        def objective(trial):
            guide_sigma=trial.suggest_float("edge_guide_sigma",.7,1.7,step=.2)
            grad_threshold=trial.suggest_float("gradient_threshold",2,30,step=2)
            dilation=trial.suggest_int("edge_dilation",0,2)
            vals=[]; deltas=[]
            for image_id,item in data.items():
                key=(image_id,guide_sigma)
                if key not in grad_cache:
                    guide=ndimage.gaussian_filter(bank[(image_id,sigma_n)].astype(float),sigma=guide_sigma,mode="reflect")
                    grad_cache[key]=np.hypot(ndimage.sobel(guide,0,mode="reflect")/8,ndimage.sobel(guide,1,mode="reflect")/8)
                edge=grad_cache[key]>grad_threshold
                if dilation: edge=ndimage.binary_dilation(edge,iterations=dilation)
                exact,joint=cache[image_id]; out=np.where(edge,exact,joint)
                score=structural_similarity(item["clean"],out,data_range=255)
                base=structural_similarity(item["clean"],exact,data_range=255)
                vals.append(score); deltas.append(score-base)
            return float(np.mean(vals)+.5*min(0.0,min(deltas)))
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=TUNE_SEED+900+ni))
        study.optimize(objective,n_trials=n_trials,show_progress_bar=False); result[str(sigma_n)]={**study.best_params,"score":study.best_value}
    return result


def evaluate(selection_csv: Path,dataset_dir: Path,out_dir: Path,payload: dict) -> pd.DataFrame:
    out_dir.mkdir(parents=True,exist_ok=True); data=load_selected(selection_csv,dataset_dir,256)
    bank=noisy_bank(data,NOISE_LEVELS,VALIDATION_SEED); rows=[]
    for image_id,item in data.items():
        clean=item["clean"]
        for sigma_n in NOISE_LEVELS:
            noisy=bank[(image_id,sigma_n)]; exact_p=payload["best"]["BF_exact"][str(sigma_n)]
            variants=[]
            start=time.perf_counter(); variants.append(("noisy",noisy,(time.perf_counter()-start)*1000,0.0))
            g,gdiag=gaussian_from_scratch(noisy,exact_p["sigma_s"],exact_p["k"],BOUNDARY); variants.append(("G",g,gdiag["runtime_ms"],1.0))
            exact,texact=apply_method("BF_exact",noisy,exact_p); variants.append(("BF_exact",exact,texact,0.0))
            for method in ("BF_skimage","JBF_gaussian_guide"):
                out,t=apply_method(method,noisy,payload["best"][method][str(sigma_n)]); variants.append((method,out,t,0.0))
            joint=variants[-1][1]; hp=payload["joint_edge_hybrid"][str(sigma_n)]; start=time.perf_counter()
            hguide=ndimage.gaussian_filter(noisy.astype(float),sigma=hp["edge_guide_sigma"],mode="reflect")
            hgrad=np.hypot(ndimage.sobel(hguide,0,mode="reflect")/8,ndimage.sobel(hguide,1,mode="reflect")/8)
            hedge=hgrad>hp["gradient_threshold"]
            if hp["edge_dilation"]: hedge=ndimage.binary_dilation(hedge,iterations=int(hp["edge_dilation"]))
            hybrid=np.where(hedge,exact,joint).astype(np.float32); hybrid_time=(time.perf_counter()-start)*1000+texact+variants[-1][2]
            variants.append(("JBF_BF_edge_hybrid",hybrid,hybrid_time,float((~hedge).mean())))
            start=time.perf_counter(); sig=noise_corrected_signal_std(noisy,sigma_n,exact_p["k"])
            mask=sig<=payload["noise_aware_gate"][str(sigma_n)]["signal_std_threshold"]
            gated=np.where(mask,g,exact).astype(np.float32); gate_time=(time.perf_counter()-start)*1000+gdiag["runtime_ms"]+texact
            variants.append(("G_BF_noise_aware",gated,gate_time,float(mask.mean())))
            ep=payload["noise_edge_gate"][str(sigma_n)]; start=time.perf_counter()
            guide=ndimage.gaussian_filter(noisy.astype(float),sigma=ep["edge_guide_sigma"],mode="reflect")
            grad=np.hypot(ndimage.sobel(guide,0,mode="reflect")/8,ndimage.sobel(guide,1,mode="reflect")/8)
            edge_mask=(sig<=ep["signal_std_threshold"])&(grad<=ep["gradient_threshold"])
            edge_gated=np.where(edge_mask,g,exact).astype(np.float32)
            edge_gate_time=(time.perf_counter()-start)*1000+gdiag["runtime_ms"]+texact
            variants.append(("G_BF_noise_edge_veto",edge_gated,edge_gate_time,float(edge_mask.mean())))
            for method,out,runtime,pg in variants:
                if method=="BF_skimage": used=payload["best"]["BF_skimage"][str(sigma_n)]
                elif method in ("JBF_gaussian_guide","JBF_BF_edge_hybrid"): used=payload["best"]["JBF_gaussian_guide"][str(sigma_n)]
                else: used=exact_p
                rows.append(dict(image_id=image_id,role=item["role"],noise_sigma=sigma_n,method=method,
                                 sigma_s=used["sigma_s"] if method!="noisy" else np.nan,
                                 sigma_r=used.get("sigma_r",np.nan) if method not in ("noisy","G") else np.nan,
                                 guide_sigma=used.get("guide_sigma",np.nan),K=used["k"] if method!="noisy" else np.nan,
                                 P_G=pg if method in ("G","G_BF_noise_aware","G_BF_noise_edge_veto") else 0.0,
                                 alternate_fraction=pg,runtime_ms=runtime,**metrics(clean,out)))
            if sigma_n==25:
                fig,axes=plt.subplots(2,4,figsize=(16,8)); axes=axes.ravel()
                for ax,(method,out,_,_) in zip(axes,variants): ax.imshow(out,cmap="gray",vmin=0,vmax=255); ax.set_title(method); ax.axis("off")
                for ax in axes[len(variants):]: ax.axis("off")
                fig.suptitle(f"{image_id}: recovery comparison, sigma_n=25"); fig.tight_layout(); fig.savefig(out_dir/f"{image_id}_recovery.png",dpi=160); plt.close(fig)
    df=pd.DataFrame(rows); df.to_csv(out_dir/"recovery_results.csv",index=False)
    agg=df.groupby(["noise_sigma","method"]).agg(PSNR=("PSNR","mean"),SSIM=("SSIM","mean"),edge=("edge_metric","mean"),
                                                      texture=("texture_metric","mean"),runtime_ms=("runtime_ms","mean"),P_G=("P_G","mean"),
                                                      alternate_fraction=("alternate_fraction","mean")).reset_index()
    agg.to_csv(out_dir/"recovery_aggregate.csv",index=False)
    return df


def append_recovery_log(df: pd.DataFrame,path: Path) -> None:
    stamp=pd.Timestamp.now(tz="UTC"); run_id=stamp.strftime("recovery_%Y%m%dT%H%M%SZ"); rows=[]
    for _,r in df.iterrows():
        rows.append(dict(timestamp=stamp.isoformat(),run_id=run_id,code_version="bf-recovery-1.0",image_id=r.image_id,
                         source_path="see final_four.csv",source_resolution="see inventory",experiment_resolution="256x256",
                         grayscale_conversion="BT.601 or original L",noise_type="additive_gaussian_clipped",noise_sigma=r.noise_sigma,
                         random_seed=VALIDATION_SEED,method=r.method,K=r.K,radius=(r.K-1)/2 if pd.notna(r.K) else np.nan,
                         sigma_s=r.sigma_s,sigma_r=r.sigma_r,epsilon=np.nan,tau=np.nan,structural_descriptor="none",
                         structural_parameters=f"guide_sigma={r.guide_sigma}" if pd.notna(r.guide_sigma) else "none",
                         h=np.nan,T_A=np.nan,MSB_depth_b=np.nan,boundary=BOUNDARY,runtime_ms=r.runtime_ms,
                         range_eval_count=np.nan,structural_eval_count=0,exponential_count=np.nan,comparison_count=np.nan,
                         P_G=r.P_G,P_BF=1-r.P_G if r.method.startswith("G_BF") else float(r.method in ("BF_exact","BF_skimage")),P_SBF=0,
                         MSE=r.MSE,PSNR=r.PSNR,SSIM=r.SSIM,edge_metric=r.edge_metric,texture_metric=r.texture_metric,
                         BF_approximation_MAE=np.nan,mean_Neff=np.nan,notes="Held-out BF recovery comparison; parameters tuned at 128"))
    append_log(rows,path)


def write_report(out_dir: Path,payload: dict,df: pd.DataFrame) -> None:
    agg=pd.read_csv(out_dir/"recovery_aggregate.csv"); best_rows=[]
    for sigma_n in NOISE_LEVELS:
        part=agg[agg.noise_sigma.eq(sigma_n)]; r=part.loc[part.SSIM.idxmax()]; best_rows.append(r)
    params=[]
    for method,levels in payload["best"].items():
        for noise,p in levels.items(): params.append(dict(method=method,noise_sigma=int(noise),**p))
    report=("# Bilateral recovery study\n\n## Why the frozen experiment failed\n\nA single sigma_r across noise 10/25/50 forced the optimizer toward a large compromise value. "
            "Raw max-min range is an extreme statistic whose value grows strongly under noise and with window size, so the deterministic tau certificate almost never fired. "
            "The new study tunes each noise level separately and evaluates on an unseen seed.\n\n## Best observed parameters\n\n"+
            dataframe_markdown(pd.DataFrame(params),4)+"\n\n## Held-out aggregate\n\n"+dataframe_markdown(agg,5)+
            "\n\n## Best SSIM method per noise level\n\n"+dataframe_markdown(pd.DataFrame(best_rows),5)+
            "\n\nThe noise-aware gate uses `sqrt(max(local_variance - sigma_n^2, 0))`; it is a statistical heuristic, not the original deterministic tau proof. "
            "Its prototype runtime includes computing both G and BF, so route fraction—not wall time—indicates potential selective savings.\n")
    (out_dir/"recovery_report.md").write_text(report)


def diagnose_sigma_tau(selection_csv: Path,dataset_dir: Path,out_dir: Path,payload: dict) -> tuple[pd.DataFrame,pd.DataFrame]:
    data=load_selected(selection_csv,dataset_dir,128); bank=noisy_bank(data,NOISE_LEVELS,VALIDATION_SEED)
    sensitivity=[]; opportunity=[]; ratios=(1,1.5,2,2.5,3,4,5,6,8)
    for sigma_n in NOISE_LEVELS:
        bp=payload["best"]["BF_exact"][str(sigma_n)]; k=bp["k"]
        for ratio in ratios:
            vals=[]
            for image_id,item in data.items():
                out,_=apply_method("BF_exact",bank[(image_id,sigma_n)],dict(sigma_s=bp["sigma_s"],sigma_r=ratio*sigma_n,k=k))
                vals.append(structural_similarity(item["clean"],out,data_range=255))
            sensitivity.append(dict(noise_sigma=sigma_n,sigma_r_over_sigma_n=ratio,sigma_r=ratio*sigma_n,
                                    sigma_s=bp["sigma_s"],K=k,SSIM_mean=float(np.mean(vals))))
        for image_id,item in data.items():
            noisy=bank[(image_id,sigma_n)]; H=ndimage.maximum_filter(noisy,size=k,mode="reflect")-ndimage.minimum_filter(noisy,size=k,mode="reflect")
            for epsilon in (.01,.05,.10):
                tau=bp["sigma_r"]*np.sqrt(-2*np.log(1-epsilon))
                opportunity.append(dict(image_id=image_id,noise_sigma=sigma_n,K=k,sigma_r=bp["sigma_r"],epsilon=epsilon,tau=tau,
                                        H_median=float(np.median(H)),H_p10=float(np.percentile(H,10)),H_p90=float(np.percentile(H,90)),
                                        exact_gate_fraction=float(np.mean(H<=tau))))
    sdf=pd.DataFrame(sensitivity); odf=pd.DataFrame(opportunity)
    sdf.to_csv(out_dir/"sigma_r_sensitivity.csv",index=False); odf.to_csv(out_dir/"tau_gate_opportunity.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for sigma_n,g in sdf.groupby("noise_sigma"): ax.plot(g.sigma_r_over_sigma_n,g.SSIM_mean,marker="o",label=f"sigma_n={sigma_n}")
    ax.set(title="Exact BF sensitivity to noise-scaled sigma_r",xlabel="sigma_r / sigma_n",ylabel="held-out mean SSIM")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out_dir/"sigma_r_sensitivity.png",dpi=160); plt.close(fig)
    return sdf,odf


def run_all(output_root: str|Path="outputs",n_trials: int=24) -> dict:
    root=Path(output_root); out=root/"07_core_ablations"/"recovery"
    payload=tune_baselines(root/"02_image_selection/final_four.csv",Path("misc"),out,n_trials)
    payload["noise_aware_gate"],payload["noise_edge_gate"]=tune_noise_aware_gate(
        root/"02_image_selection/final_four.csv",Path("misc"),payload["best"],n_trials)
    payload["joint_edge_hybrid"]=tune_joint_edge_hybrid(
        root/"02_image_selection/final_four.csv",Path("misc"),payload["best"],max(n_trials,32))
    (out/"recovery_best_parameters.json").write_text(json.dumps(payload,indent=2))
    df=evaluate(root/"02_image_selection/final_four.csv",Path("misc"),out,payload); write_report(out,payload,df)
    append_recovery_log(df,root/"reports"/"experiment_log.csv")
    diagnose_sigma_tau(root/"02_image_selection/final_four.csv",Path("misc"),out,payload)
    return dict(parameters=payload,results=df)


if __name__=="__main__":
    result=run_all(); print(json.dumps(result["parameters"],indent=2))
    print(pd.read_csv("outputs/07_core_ablations/recovery/recovery_aggregate.csv").to_string(index=False))

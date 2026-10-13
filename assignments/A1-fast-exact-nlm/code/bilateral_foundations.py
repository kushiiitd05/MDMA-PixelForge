"""From-scratch spatial-domain Gaussian and bilateral filtering foundations.

The implementation is intentionally explicit about p, q, odd centered support,
boundary extension, normalization, operation counts, and reproducible logging.
"""

from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_bilateral
from skimage.transform import resize

from bilateral_dataset_pipeline import lbp8, read_image


CODE_VERSION = "foundation-v1.1"
DEFAULT_SEED = 20260826
DEFAULT_SIGMA_S = 1.0
DEFAULT_SIGMA_R = 30.0
DEFAULT_K = 7
BOUNDARY = "reflect"


def dataframe_markdown(df: pd.DataFrame, float_digits: int = 6) -> str:
    """Small dependency-free Markdown table formatter."""
    shown=df.copy()
    for col in shown.select_dtypes(include=[np.number]).columns:
        shown[col]=shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.{float_digits}f}")
    headers=[str(c) for c in shown.columns]
    lines=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    for row in shown.astype(str).itertuples(index=False,name=None):
        lines.append("| "+" | ".join(v.replace("|","\\|") for v in row)+" |")
    return "\n".join(lines)


def validate_window(k: int) -> int:
    if not isinstance(k, (int, np.integer)) or k < 1 or k % 2 == 0:
        raise ValueError("K must be a positive odd integer so the window has one unique center pixel")
    return k // 2


def recommended_k(sigma_s: float) -> int:
    if sigma_s <= 0:
        raise ValueError("sigma_s must be positive")
    return 2 * math.ceil(3 * sigma_s) + 1


def _np_pad_mode(boundary: str) -> str:
    # scipy reflect = half-sample symmetric = numpy symmetric.
    # scipy mirror  = whole-sample symmetric = numpy reflect.
    try:
        return {"reflect": "symmetric", "mirror": "reflect"}[boundary]
    except KeyError as exc:
        raise ValueError("boundary must be 'reflect' or 'mirror'") from exc


def spatial_kernel(k: int, sigma_s: float) -> np.ndarray:
    radius = validate_window(k)
    if sigma_s <= 0:
        raise ValueError("sigma_s must be positive")
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    return np.exp(-(xx*xx + yy*yy) / (2.0 * sigma_s*sigma_s)).astype(np.float64)


def gaussian_from_scratch(image: np.ndarray, sigma_s: float = DEFAULT_SIGMA_S,
                          k: int | None = None, boundary: str = BOUNDARY) -> tuple[np.ndarray, dict]:
    """Centered 2-D Gaussian weighted average, evaluated directly in the pixel domain."""
    k = recommended_k(sigma_s) if k is None else k
    radius = validate_window(k); gs = spatial_kernel(k, sigma_s); denom = float(gs.sum())
    x = np.asarray(image, dtype=np.float64)
    padded = np.pad(x, radius, mode=_np_pad_mode(boundary))
    windows = sliding_window_view(padded, (k, k))
    start = time.perf_counter()
    out = np.einsum("ijkl,kl->ij", windows, gs, optimize=True) / denom
    runtime_ms = (time.perf_counter() - start) * 1000.0
    n = x.size; q = k*k
    ops = dict(method="G", K=k, radius=radius, sigma_s=sigma_s, sigma_r=np.nan,
               runtime_ms=runtime_ms, range_eval_count=0, structural_eval_count=0,
               exponential_count=0, comparison_count=0,
               multiplication_count=n*q, addition_count=n*(q-1), division_count=n)
    return out.astype(np.float32), ops


def bilateral_from_scratch(image: np.ndarray, sigma_s: float = DEFAULT_SIGMA_S,
                           sigma_r: float = DEFAULT_SIGMA_R, k: int | None = None,
                           boundary: str = BOUNDARY, chunk_rows: int = 32,
                           return_neff: bool = False) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    """Exact local BF: each q gets its own Gs(p,q)*Gr(p,q), then one normalization."""
    k = recommended_k(sigma_s) if k is None else k
    radius = validate_window(k)
    if sigma_r <= 0:
        raise ValueError("sigma_r must be positive")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    gs = spatial_kernel(k, sigma_s)
    x = np.asarray(image, dtype=np.float64)
    padded = np.pad(x, radius, mode=_np_pad_mode(boundary))
    windows = sliding_window_view(padded, (k, k))
    out = np.empty_like(x); neff = np.empty_like(x) if return_neff else None
    inv_two_sr2 = 1.0 / (2.0*sigma_r*sigma_r)
    start = time.perf_counter()
    for y0 in range(0, x.shape[0], chunk_rows):
        y1 = min(y0+chunk_rows, x.shape[0])
        win = windows[y0:y1]
        center = x[y0:y1, :, None, None]
        delta = center - win
        gr = np.exp(-(delta*delta) * inv_two_sr2)
        weights = gr * gs
        denom = weights.sum(axis=(-2,-1))
        out[y0:y1] = (weights*win).sum(axis=(-2,-1)) / denom
        if return_neff:
            neff[y0:y1] = denom*denom / np.maximum((weights*weights).sum(axis=(-2,-1)), 1e-15)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    n=x.size; q=k*k
    ops=dict(method="BF",K=k,radius=radius,sigma_s=sigma_s,sigma_r=sigma_r,
             runtime_ms=runtime_ms,range_eval_count=n*q,structural_eval_count=0,
             exponential_count=n*q,comparison_count=0,
             subtraction_count=n*q,multiplication_count=n*(4*q+1),
             addition_count=n*(2*(q-1)),division_count=2*n)
    if return_neff:
        return out.astype(np.float32), ops, neff.astype(np.float32)
    return out.astype(np.float32), ops


def bilateral_scalar_reference(image: np.ndarray, sigma_s: float, sigma_r: float,
                               k: int, boundary: str = BOUNDARY) -> np.ndarray:
    """Slow independent loop reference used only for tiny validation arrays."""
    radius=validate_window(k); gs=spatial_kernel(k,sigma_s); x=np.asarray(image,dtype=np.float64)
    padded=np.pad(x,radius,mode=_np_pad_mode(boundary)); out=np.empty_like(x)
    for yp in range(x.shape[0]):
        for xp in range(x.shape[1]):
            numerator=0.0; denominator=0.0; center=float(x[yp,xp])
            for ky,dy in enumerate(range(-radius,radius+1)):
                for kx,dx in enumerate(range(-radius,radius+1)):
                    iq=float(padded[yp+radius+dy,xp+radius+dx])
                    gr=math.exp(-((center-iq)**2)/(2*sigma_r*sigma_r)); w=float(gs[ky,kx])*gr
                    numerator += w*iq; denominator += w
            out[yp,xp]=numerator/denominator
    return out.astype(np.float32)


def pq_dry_run(image: np.ndarray, p: tuple[int,int], sigma_s: float, sigma_r: float,
               k: int, boundary: str = BOUNDARY) -> tuple[pd.DataFrame, dict]:
    """Return one auditable row per q for the requested reference pixel p=(y,x)."""
    radius=validate_window(k); gs=spatial_kernel(k,sigma_s); x=np.asarray(image,dtype=np.float64)
    yp,xp=p
    if not (0<=yp<x.shape[0] and 0<=xp<x.shape[1]): raise IndexError("p is outside the image")
    padded=np.pad(x,radius,mode=_np_pad_mode(boundary)); center=float(x[yp,xp]); rows=[]
    for ky,dy in enumerate(range(-radius,radius+1)):
        for kx,dx in enumerate(range(-radius,radius+1)):
            iq=float(padded[yp+radius+dy,xp+radius+dx]); d2=dx*dx+dy*dy
            delta=center-iq; gr=math.exp(-(delta*delta)/(2*sigma_r*sigma_r)); gsv=float(gs[ky,kx]); w=gsv*gr
            rows.append(dict(q_y=yp+dy,q_x=xp+dx,dx=dx,dy=dy,d_squared=d2,I_p=center,I_q=iq,
                             delta_I=delta,Gs=gsv,Gr=gr,Gt=1.0,combined_w=w,w_times_Iq=w*iq))
    table=pd.DataFrame(rows)
    sums=dict(sum_wIq=float(table.w_times_Iq.sum()),sum_w=float(table.combined_w.sum()))
    sums["O_p"]=sums["sum_wIq"]/sums["sum_w"]
    return table,sums


def synthetic_cases(size: int = 21) -> dict[str,np.ndarray]:
    if size%2==0 or size<7: raise ValueError("synthetic size must be odd and >=7")
    flat=np.full((size,size),100.0,dtype=np.float32)
    step=np.full((size,size),50.0,dtype=np.float32); step[:,size//2:]=200.0
    ramp=np.tile(np.linspace(20,235,size,dtype=np.float32),(size,1))
    periodic=np.tile(np.where(np.arange(size)%2==0,100.0,102.0),(size,1)).astype(np.float32)
    conflict=np.full((size,size),100.0,dtype=np.float32)
    yy,xx=np.mgrid[:size,:size]; mask=(abs(yy-size//2)<=4)&(abs(xx-size//2)<=4)
    conflict[mask]=100.0+2.0*((xx[mask]+yy[mask])%2)
    conflict[:,3]=180.0
    return dict(flat=flat,step_edge=step,linear_ramp=ramp,low_amplitude_periodic=periodic,
                intensity_similar_structural_conflict=conflict)


def gradient_correlation(reference: np.ndarray, test: np.ndarray) -> float:
    def mag(x): return np.hypot(ndimage.sobel(x,0,mode="reflect"),ndimage.sobel(x,1,mode="reflect"))
    a,b=mag(reference).ravel(),mag(test).ravel()
    if a.std()==0 or b.std()==0: return float(a.std()==b.std()==0)
    return float(np.corrcoef(a,b)[0,1])


def lbp_hist_similarity(reference: np.ndarray, test: np.ndarray) -> float:
    a=np.bincount(lbp8(reference).ravel(),minlength=256).astype(float)
    b=np.bincount(lbp8(test).ravel(),minlength=256).astype(float); a/=a.sum(); b/=b.sum()
    return float(np.minimum(a,b).sum())


def metrics(clean: np.ndarray, output: np.ndarray) -> dict:
    return dict(MSE=float(mean_squared_error(clean,output)),
                PSNR=float(peak_signal_noise_ratio(clean,output,data_range=255)),
                SSIM=float(structural_similarity(clean,output,data_range=255)),
                edge_metric=gradient_correlation(clean,output),
                texture_metric=lbp_hist_similarity(clean,output))


def validate_foundations(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True,exist_ok=True); cases=synthetic_cases(); validations=[]
    # Odd-window and invariance tests.
    try: gaussian_from_scratch(cases["flat"],k=4); even_rejected=False
    except ValueError: even_rejected=True
    g,gops=gaussian_from_scratch(cases["flat"],sigma_s=1,k=7)
    bf,bops=bilateral_from_scratch(cases["flat"],sigma_s=1,sigma_r=30,k=7)
    validations += [dict(test="even_K_rejected",value=float(even_rejected),tolerance=1,passed=even_rejected),
                    dict(test="gaussian_flat_invariance",value=float(np.max(abs(g-100))),tolerance=1e-5,passed=np.max(abs(g-100))<1e-5),
                    dict(test="bilateral_flat_invariance",value=float(np.max(abs(bf-100))),tolerance=1e-5,passed=np.max(abs(bf-100))<1e-5)]
    rng=np.random.default_rng(7); tiny=rng.uniform(0,255,(9,11)).astype(np.float32)
    g,_=gaussian_from_scratch(tiny,1,7,"reflect")
    gsci=ndimage.gaussian_filter(tiny.astype(float),sigma=1,radius=3,mode="reflect")
    err=float(np.max(abs(g-gsci))); validations.append(dict(test="gaussian_vs_scipy_max_abs",value=err,tolerance=2e-5,passed=err<2e-5))
    bf,_=bilateral_from_scratch(tiny,1,30,5,"reflect")
    scalar=bilateral_scalar_reference(tiny,1,30,5,"reflect")
    err=float(np.max(abs(bf-scalar))); validations.append(dict(test="bilateral_vectorized_vs_scalar_max_abs",value=err,tolerance=2e-5,passed=err<2e-5))
    gstep,_=gaussian_from_scratch(cases["step_edge"],1,7); bstep,_=bilateral_from_scratch(cases["step_edge"],1,30,7)
    c=cases["step_edge"].shape[1]//2
    gerr=float(np.mean(abs(gstep[:,c-1:c+1]-cases["step_edge"][:,c-1:c+1])))
    berr=float(np.mean(abs(bstep[:,c-1:c+1]-cases["step_edge"][:,c-1:c+1])))
    validations.append(dict(test="BF_preserves_step_better_than_G",value=berr/max(gerr,1e-12),tolerance=1.0,passed=berr<gerr))
    # Independent LUT-accelerated library cross-check on the canonical edge case.
    # skimage uses numpy padding names, so 'symmetric' matches this project's half-sample 'reflect'.
    sk=denoise_bilateral(cases["step_edge"]/255.0,win_size=7,sigma_color=30/255.0,sigma_spatial=1,
                         bins=10000,mode="symmetric",channel_axis=None)*255.0
    sk_mae=float(np.mean(abs(bstep-sk)))
    validations.append(dict(test="bilateral_step_vs_skimage_MAE",value=sk_mae,tolerance=.01,passed=sk_mae<.01))
    blarge,_=bilateral_from_scratch(tiny,1,1e6,7); limit=float(np.max(abs(blarge-g)))
    validations.append(dict(test="BF_large_sigma_r_approaches_G",value=limit,tolerance=2e-5,passed=limit<2e-5))
    vdf=pd.DataFrame(validations); vdf.to_csv(out_dir/"validation_results.csv",index=False)
    dry_image=np.array([[50,50,50,200,200],[50,50,50,200,200],[50,50,50,200,200],
                        [50,50,50,200,200],[50,50,50,200,200]],dtype=np.float32)
    table,sums=pq_dry_run(dry_image,(2,2),1,30,3); table.to_csv(out_dir/"pq_dry_run.csv",index=False)
    fig,axes=plt.subplots(len(cases),3,figsize=(9,3*len(cases)))
    for row,(name,case) in enumerate(cases.items()):
        cg,_=gaussian_from_scratch(case,1,7); cb,_=bilateral_from_scratch(case,1,30,7)
        for ax,data,title in zip(axes[row],(case,cg,cb),(f"{name}: input","Gaussian","BF sigma_r=30")):
            ax.imshow(data,cmap="gray",vmin=0,vmax=255); ax.set_title(title,fontsize=9); ax.axis("off")
    fig.suptitle("Required synthetic foundation cases"); fig.tight_layout(); fig.savefig(out_dir/"synthetic_case_comparison.png",dpi=160); plt.close(fig)
    mid=cases["step_edge"].shape[0]//2
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(cases["step_edge"][mid],label="input",lw=2)
    ax.plot(gstep[mid],label="Gaussian"); ax.plot(bstep[mid],label="BF")
    ax.set(title="Step-edge spatial profile",xlabel="x",ylabel="intensity",ylim=(0,255)); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out_dir/"step_edge_profile.png",dpi=160); plt.close(fig)
    md=["# Foundation validation report","",f"Passed **{int(vdf.passed.sum())}/{len(vdf)}** checks.","",
        "Exact numerical BF equivalence is tested against an independent scalar loop. The scikit-image check is a trusted, LUT-accelerated cross-check on the canonical step edge; arbitrary-image pixel equality is not asserted because library support/LUT conventions differ.","",
        dataframe_markdown(vdf),"","## Mandatory p,q dry-run","",
        "Reference pixel `p=(y=2,x=2)`, K=3, sigma_s=1, sigma_r=30. `Gt=1` because no structural cue is active in classical BF.","",
        dataframe_markdown(table),"",
        f"- sum(w*I(q)) = {sums['sum_wIq']:.9f}",f"- sum(w) = {sums['sum_w']:.9f}",f"- BF(p) = {sums['O_p']:.9f}"]
    (out_dir/"validation_report.md").write_text("\n".join(md))
    if not bool(vdf.passed.all()): raise AssertionError("Foundation validation failed; inspect validation_results.csv")
    return dict(results=vdf,dry_table=table,dry_sums=sums,cases=cases)


LOG_COLUMNS=["timestamp","run_id","code_version","code_sha256","image_id","source_path","source_resolution",
             "experiment_resolution","grayscale_conversion","noise_type","noise_sigma","random_seed","method","K",
             "radius","sigma_s","sigma_r","epsilon","tau","structural_descriptor","structural_parameters","h","T_A",
             "MSB_depth_b","boundary","runtime_ms","range_eval_count","structural_eval_count","exponential_count",
             "comparison_count","multiplication_count","addition_count","division_count","P_G","P_BF","P_SBF","MSE",
             "PSNR","SSIM","edge_metric","texture_metric","BF_approximation_MAE","mean_Neff","notes"]


def append_log(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows)
    for col in LOG_COLUMNS:
        if col not in df: df[col]=np.nan
    df=df[LOG_COLUMNS]
    df.to_csv(path,mode="a",header=not path.exists(),index=False)


def write_foundation_report(sdf: pd.DataFrame, out_dir: Path) -> None:
    view=sdf[["image_id","source_path","method","sigma_r","PSNR","SSIM","edge_metric",
              "texture_metric","mean_Neff","runtime_ms"]].copy()
    view["setting"]=np.where(view.method.eq("G"),"G",view.sigma_r.map(lambda x:f"BF sigma_r={x:g}"))
    view=view[["image_id","source_path","setting","PSNR","SSIM","edge_metric","texture_metric","mean_Neff","runtime_ms"]]
    best=sdf.loc[sdf.groupby("image_id").PSNR.idxmax(),["image_id","method","sigma_r","PSNR","SSIM"]].copy()
    best["setting"]=np.where(best.method.eq("G"),"G",best.sigma_r.map(lambda x:f"BF sigma_r={x:g}"))
    text=("# Selected-image Gaussian/BF foundation results\n\n## All settings\n\n"+
          dataframe_markdown(view,4)+"\n\n## Best PSNR inside this initial grid\n\n"+
          dataframe_markdown(best[["image_id","setting","PSNR","SSIM"]],4)+
          "\n\nThese are seeded sigma_n=25 foundation measurements, not final tuned claims. "
          "The three natural/smooth representatives favor Gaussian within sigma_r<=50, while the dense ruler edges favor BF. "
          "Therefore the next parameter study must extend sigma_r rather than labeling this initial grid optimal. "
          "Log/gamma/stretch probes are not used as inputs.\n")
    (out_dir/"foundation_results.md").write_text(text)


def run_selected_foundations(selection_csv: Path, dataset_dir: Path, out_dir: Path,
                             experiment_log: Path, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    out_dir.mkdir(parents=True,exist_ok=True); selected=pd.read_csv(selection_csv); rng=np.random.default_rng(seed)
    rows=[]; summary=[]; code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); run_id=datetime.now(timezone.utc).strftime("foundation_%Y%m%dT%H%M%SZ")
    for _,r in selected.iterrows():
        _,native,_=read_image(dataset_dir/r.relative_path)
        if min(native.shape)>256:
            clean=resize(native,(256,256),order=1,mode="reflect",anti_aliasing=True,preserve_range=True).astype(np.float32)
        else: clean=native.astype(np.float32)
        noise=rng.normal(0,25,clean.shape); noisy=np.clip(clean+noise,0,255).astype(np.float32)
        methods=[]
        gout,gops=gaussian_from_scratch(noisy,DEFAULT_SIGMA_S,DEFAULT_K,BOUNDARY)
        gs=spatial_kernel(DEFAULT_K,DEFAULT_SIGMA_S); g_neff=float(gs.sum()**2/(gs*gs).sum())
        methods.append(("G",gout,gops,g_neff))
        for sr in (10.0,20.0,30.0,50.0):
            bout,bops,neff=bilateral_from_scratch(noisy,DEFAULT_SIGMA_S,sr,DEFAULT_K,BOUNDARY,return_neff=True)
            methods.append(("BF",bout,bops,float(neff.mean())))
        fig,axes=plt.subplots(2,3,figsize=(12,8)); axes=axes.ravel()
        axes[0].imshow(clean,cmap="gray",vmin=0,vmax=255); axes[0].set_title("clean")
        axes[1].imshow(noisy,cmap="gray",vmin=0,vmax=255); axes[1].set_title("noisy sigma_n=25")
        for ax,(name,out,ops,_) in zip(axes[2:],methods[:4]):
            label="Gaussian" if name=="G" else f"BF sigma_r={ops['sigma_r']:g}"
            ax.imshow(out,cmap="gray",vmin=0,vmax=255); ax.set_title(label)
        for ax in axes: ax.axis("off")
        fig.suptitle(f"{r.image_id} {r.filename}: foundation outputs"); fig.tight_layout(); fig.savefig(out_dir/f"{r.image_id}_foundation.png",dpi=150); plt.close(fig)
        for name,out,ops,mean_neff in methods:
            met=metrics(clean,out); now=datetime.now(timezone.utc).isoformat()
            record=dict(timestamp=now,run_id=run_id,code_version=CODE_VERSION,code_sha256=code_hash,image_id=r.image_id,
                        source_path=r.relative_path,source_resolution=f"{int(r.height)}x{int(r.width)}",
                        experiment_resolution=f"{clean.shape[0]}x{clean.shape[1]}",grayscale_conversion="BT.601 or original L",
                        noise_type="additive_gaussian_clipped",noise_sigma=25,random_seed=seed,
                        epsilon=np.nan,tau=np.nan,structural_descriptor="none",structural_parameters="none",h=np.nan,T_A=np.nan,
                        MSB_depth_b=np.nan,boundary=BOUNDARY,P_G=1.0 if name=="G" else 0.0,
                        P_BF=1.0 if name=="BF" else 0.0,P_SBF=0.0,BF_approximation_MAE=0.0 if name=="BF" else np.nan,
                        mean_Neff=mean_neff,notes="Foundation sweep; every 256 copy derived directly from native source",**ops,**met)
            rows.append(record); summary.append(record)
    append_log(rows,experiment_log)
    sdf=pd.DataFrame(summary); sdf.to_csv(out_dir/"foundation_summary.csv",index=False)
    write_foundation_report(sdf,out_dir)
    return sdf


def run_all(output_root: str | Path = "outputs") -> dict:
    root=Path(output_root); foundation=root/"04_gaussian_bf_foundation"; synthetic=root/"05_synthetic_proofs"
    validation=validate_foundations(synthetic)
    summary=run_selected_foundations(root/"02_image_selection/final_four.csv",Path("misc"),foundation,
                                     root/"reports/experiment_log.csv")
    return dict(validation=validation,summary=summary)


if __name__ == "__main__":
    result=run_all()
    print(result["validation"]["results"].to_string(index=False))
    print("\nFoundation PSNR/SSIM:")
    print(result["summary"][["image_id","method","sigma_r","PSNR","SSIM","runtime_ms"]].to_string(index=False))

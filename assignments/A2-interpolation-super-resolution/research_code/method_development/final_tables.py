"""Final tables, source-aware paired statistics, trade-off plots and the deployment manifest (no training, no selection).

Reads results/final_eval/{per_image.jsonl, latency.json, roles_frozen.json}, results/final_scales/*.json.
Writes results/final_eval/: per_image.csv, method_summary.csv, role_table.csv/.md, paired.csv, seeds.csv, residuals.csv,
fig_tradeoff_val.png, fig_tradeoff_urban.png; submission/extension/{weights/, model_manifest.json, results/}.
Paired intervals: 10k bootstrap over source units (urban100c 100 images, div2k_eval20 20 images), pointwise and
descriptive (no multiplicity adjustment). fresh: per-source means and leave-one-source-out range only (3 sources).
assignment: per-image values only (4 images, repeatedly viewed during development).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MD = Path(__file__).resolve().parent
RES = MD / "results" / "final_eval"
SUB = MD.parent / "submission" / "extension"
EXT_SETS = ("assignment", "fresh", "div2k_eval20", "urban100c")
ROLE_NAMES = {"psnr": "Highest PSNR", "ssim": "Highest SSIM", "fastest": "Lowest latency", "tradeoff": "Best trade-off (promoted)"}


def load() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.DataFrame([json.loads(l) for l in (RES / "per_image.jsonl").read_text().splitlines()])
    # all scored rows come from the GPU run of final_eval.py (CUDA); the "MPS" wording in impl strings reflects the code path, not the device used
    for col in ("impl", "projection_impl"):
        df[col] = df[col].astype(str).str.replace("torch float32 MPS", "torch float32 (GPU CUDA)").str.replace("torch MPS float32", "torch float32 (GPU CUDA)").str.replace("torch MPS bicubic", "torch bicubic (GPU CUDA)")
    assert not df.duplicated(["stage", "dataset", "scale", "method_id", "image_id"]).any(), "duplicate per-image rows"
    lat = json.loads((RES / "latency.json").read_text())
    roles = json.loads((RES / "roles_frozen.json").read_text())
    return df, lat, roles


def lat_of(lat: dict, s: int, mid: str) -> dict:
    sys.path.insert(0, str(MD))
    from select_roles import latency_key
    return lat[latency_key(s, mid)]


def label(mid: str) -> str:
    return (mid.replace("classical:", "").replace("pretrained:", "").replace("ours:", "OURS ").replace("finetuned:", "").replace("+proj", " + exact proj.")
            .replace("fs_x2_", "").replace("fs_x4_", "").replace("fs_x8_", "").replace("_psnr", " [val-PSNR ckpt]").replace("_ssim", " [val-SSIM ckpt]"))


def short(mid: str) -> str:
    m = mid.split(":", 1)[1]
    proj = " +P" if m.endswith("+proj") else ""
    m = m.replace("+proj", "")
    if m.startswith("fs_x"):
        _, _, pool, seed, crit = m.split("_")
        return f"ours {seed} {crit}{proj}"
    if m.startswith("swinirP1"):
        return f"SwinIR-M P1-ft {m.split('_')[-1]}{proj}"
    return m.replace("final_bal_", "old core ") + proj


def boot(d: np.ndarray, n: int = 10000) -> list[float]:
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def paired(ext: pd.DataFrame, s: int, a: str, b: str) -> list[dict]:
    rows = []
    for ds in EXT_SETS:
        x = ext[(ext.scale == s) & (ext.dataset == ds)]
        m = x[x.method_id == a].merge(x[x.method_id == b], on=["image_id", "source_id"], suffixes=("_a", "_b"))
        if not len(m):
            continue
        n_exp = x[x.method_id == a].image_id.nunique()
        assert len(m) == n_exp, (s, ds, a, b, len(m), n_exp)
        for metric in ("psnr_db", "ssim"):
            d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
            r = {"scale": s, "dataset": ds, "a": a, "b": b, "metric": metric, "n_images": len(d), "mean_diff": round(float(d.mean()), 4), "a_better": f"{int((d > 0).sum())}/{len(d)}"}
            if ds in ("urban100c", "div2k_eval20"):
                r["boot95_over_sources_pointwise"] = boot(d)
            elif ds == "fresh":
                per = pd.Series(d, index=m.source_id.values).groupby(level=0).mean()
                r["per_source_mean"] = {k: round(float(v), 4) for k, v in per.items()}
                r["leave_one_source_out_range"] = [round(float(per.drop(k).mean()), 4) for k in per.index]
                r["leave_one_source_out_range"] = [min(r["leave_one_source_out_range"]), max(r["leave_one_source_out_range"])]
            else:
                r["per_image"] = {k: round(float(v), 4) for k, v in zip(m.image_id, d)}
            rows.append(r)
    return rows


def best_in(cands: list[dict], family: str) -> str:
    c = [x for x in cands if x["family"] == family]
    return sorted(c, key=lambda x: (-x["p"], -x["q"], x["t"]))[0]["method_id"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    df, lat, roles = load()
    df.to_csv(RES / "per_image.csv", index=False)
    val, ext = df[df.stage == "val"], df[df.stage == "external"]
    summ = df.groupby(["stage", "dataset", "scale", "method_id", "family"]).agg(psnr=("psnr_db", "mean"), ssim=("ssim", "mean"), n=("image_id", "nunique")).reset_index()
    summ.to_csv(RES / "method_summary.csv", index=False)
    S = lambda st, ds, s, mid, col: float(summ[(summ.stage == st) & (summ.dataset == ds) & (summ.scale == s) & (summ.method_id == mid)][col].iloc[0]) if len(summ[(summ.stage == st) & (summ.dataset == ds) & (summ.scale == s) & (summ.method_id == mid)]) else float("nan")

    role_rows, pair_rows, resid_rows, seed_rows = [], [], [], []
    for s in (2, 4, 8):
        R = roles["scales"][str(s)]
        cands = R["candidates"]
        info = {c["method_id"]: c for c in cands}
        for scope, rset in (("overall", R["roles"]), ("ours", R["ours_best"])):
            for role in ("psnr", "ssim", "fastest", "tradeoff"):
                mid = rset.get(role)
                if mid is None:
                    role_rows.append({"scale": s, "scope": scope, "role": ROLE_NAMES[role], "method_id": None, "reason": rset.get("tradeoff_rule")})
                    continue
                c, L = info[mid], lat_of(lat, s, mid)
                reason = {"psnr": "max val PSNR", "ssim": "max val SSIM", "fastest": "min median latency", "tradeoff": rset.get("tradeoff_rule")}[role]
                row = {"scale": s, "scope": scope, "role": ROLE_NAMES[role], "method_id": mid, "method": label(mid), "family": c["family"],
                       "val_psnr": round(c["p"], 3), "val_ssim": round(c["q"], 4), "latency_median_ms": round(L["median_of_medians_ms"], 3), "latency_iqr_ms": round(L["max_iqr_ms"], 3),
                       "x_torch_bicubic": round(c["t_ratio"], 2), "reason": reason}
                for ds in EXT_SETS:
                    row[f"{ds}_psnr"], row[f"{ds}_ssim"] = round(S("external", ds, s, mid, "psnr"), 3), round(S("external", ds, s, mid, "ssim"), 4)
                p = df[(df.method_id == mid) & (df.scale == s)]
                row["params"] = int(p.params.dropna().iloc[0]) if "params" in p and p.params.notna().any() else None
                row["checkpoint_id"] = p.checkpoint_id.dropna().iloc[0] if p.checkpoint_id.notna().any() else None
                role_rows.append(row)
        # paired comparisons for the promoted pipeline and the PSNR winner of our family
        promo, ours_p = R["roles"]["tradeoff"], R["ours_best"]["psnr"]
        refs = {"strongest classical (val)": best_in(cands, "classical"), "strongest pretrained (val)": best_in(cands, "pretrained")}
        if any(c["family"] == "finetuned" for c in cands):
            refs["strongest P1-fine-tuned SwinIR-M (val)"] = best_in(cands, "finetuned")
        if s == 8:
            refs["previous 8x core"] = "ours:final_bal_s7+proj"
        for a in dict.fromkeys([promo, ours_p]):
            for tag, b in refs.items():
                if a != b and b in info:
                    for r in paired(ext, s, a, b):
                        pair_rows.append({**r, "comparison": tag})
        pr = ext[(ext.scale == s) & ext.method_id.str.endswith("+proj") & ext.lr_resid_max_preclip.notna() & ~ext.method_id.str.contains("ibp20")]  # IBP is oblique, not exact
        resid_rows += pr.groupby("method_id").agg(max_preclip=("lr_resid_max_preclip", "max"), rms_postclip_mean=("lr_resid_rms_postclip", "mean")).reset_index().assign(scale=s).to_dict(orient="records")
        for seed in sorted(set(ext[(ext.scale == s) & (ext.family == "ours")].seed.dropna())):
            for mid in sorted(set(ext[(ext.scale == s) & (ext.seed == seed) & (ext.family == "ours") & ext.method_id.str.endswith("+proj")].method_id)):
                seed_rows.append({"scale": s, "seed": int(seed), "method_id": mid, **{ds: round(S("external", ds, s, mid, "psnr"), 3) for ds in EXT_SETS}, "val": round(S("val", "val", s, mid, "psnr"), 3)})

    rt = pd.DataFrame(role_rows)
    rt.to_csv(RES / "role_table.csv", index=False)
    (RES / "role_table.md").write_text(rt.to_markdown(index=False))
    pd.DataFrame(pair_rows).to_json(RES / "paired.json", orient="records", indent=1)
    pd.DataFrame(pair_rows).drop(columns=[c for c in ("per_image", "per_source_mean") if c in pd.DataFrame(pair_rows)], errors="ignore").to_csv(RES / "paired.csv", index=False)
    pd.DataFrame(resid_rows).to_csv(RES / "residuals.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(RES / "seeds.csv", index=False)
    plots(summ, lat, roles)
    package(df, roles)
    print(rt[rt.scope == "overall"][["scale", "role", "method", "val_psnr", "val_ssim", "latency_median_ms", "urban100c_psnr", "urban100c_ssim", "assignment_psnr"]].to_string())


def plots(summ: pd.DataFrame, lat: dict, roles: dict) -> None:
    colors = {"classical": "#6b7280", "pretrained": "#b45309", "finetuned": "#15803d", "ours": "#1d4ed8"}
    for stage, ds, fname in (("val", "val", "fig_tradeoff_val.png"), ("external", "urban100c", "fig_tradeoff_urban.png")):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        for j, s in enumerate((2, 4, 8)):
            x = summ[(summ.stage == stage) & (summ.dataset == ds) & (summ.scale == s)]
            R = roles["scales"][str(s)]["roles"]
            show = {R["psnr"]: "PSNR role", R["ssim"]: "SSIM role", R["fastest"]: "fastest", R["tradeoff"]: "promoted"}
            for fam in ("classical", "pretrained", "finetuned", "ours"):
                f = x[x.family == fam]
                if len(f):
                    show.setdefault(f.sort_values("psnr").iloc[-1].method_id, f"best {fam}")
            for i, (col, yl) in enumerate((("psnr", "mean PSNR (dB)"), ("ssim", "mean SSIM"))):
                ax = axes[i, j]
                k = 0
                for _, r in x.iterrows():
                    t = lat_of(lat, s, r.method_id)["median_of_medians_ms"]
                    promo = r.method_id == R["tradeoff"]
                    ax.scatter(t, r[col], s=90 if promo else 22, c=colors[r.family], marker="*" if promo else "o", zorder=3 if promo else 2,
                               edgecolors="k" if promo else "none")
                    if r.method_id in show:
                        ax.annotate(f"{show[r.method_id]}: {short(r.method_id)}", (t, r[col]), fontsize=5.5, xytext=(4, -9 - 7 * (k % 3) if k % 2 else 4 + 7 * (k % 3)),
                                    textcoords="offset points", arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.4"})
                        k += 1
                bic = roles["scales"][str(s)]["bicubic_ref_ms"]
                ax.margins(y=0.18)
                ax.axvline(10 * bic, color="r", lw=0.8, ls="--")
                ax.set_xscale("log")
                ax.set_title(f"{s}x ({512 // s}² → 512²) · {ds}", fontsize=9)
                ax.set_ylabel(yl, fontsize=8)
                ax.set_xlabel("median full decode latency (ms, log) — red: 10× torch bicubic budget", fontsize=7)
                ax.grid(alpha=0.3)
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()] + [plt.Line2D([], [], marker="*", ls="", color="k", markersize=10, label="promoted trade-off")]
        fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8)
        fig.text(0.5, 0.005, "Apple M2 Pro 8 GB, torch MPS / numpy CPU, host-to-host, isolated processes; HAT-L uses memory-forced tiling (machine-specific).", ha="center", fontsize=7)
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        fig.savefig(RES / fname, dpi=160)
        plt.close(fig)


def package(df: pd.DataFrame, roles: dict) -> None:
    """Copy only the selected deployment weights (our family's promoted + PSNR + SSIM checkpoints) with hashes."""
    (SUB / "weights").mkdir(parents=True, exist_ok=True)
    (SUB / "results").mkdir(parents=True, exist_ok=True)
    fs = MD / "results" / "final_scales"
    man = {"method": "x_hat = B y + P F(y); F = DirectLR(scale, width 96, depth 12) with bicubic skip; exact P1 projection; clip [0,255]",
           "input_normalisation": "y/255 float32; output *255", "degradation": "P1_gauss_rec (sigma=1 per x2 stage, half-pixel lattice, symmetric borders)",
           "selection": "validation images 4.2.01, 7.1.05, boat.512, 5.2.09 only; roles by select_roles.py (frozen before external scoring)",
           "environment": "conda eda311, torch MPS, Apple M2 Pro", "reproduce_training": "python method_development/run_final_scales.py --scale S --pool POOL --seeds SEED",
           "models": {}}
    for s in (2, 4, 8):
        R = roles["scales"][str(s)]
        man["models"][str(s)] = {}
        for role, mid in (("promoted", R["ours_best"]["tradeoff"]), ("psnr", R["ours_best"]["psnr"]), ("ssim", R["ours_best"]["ssim"])):
            if mid is None:
                continue
            r = df[(df.method_id == mid) & (df.scale == s)].iloc[0]
            src = MD / "checkpoints" / r.checkpoint_id
            dst = SUB / "weights" / r.checkpoint_id
            if not dst.exists():
                shutil.copy2(src, dst)
            info_path = fs / f"x{s}_{r.training_pool_id}_s{int(r.seed)}.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            man["models"][str(s)][role] = {"method_id": mid, "weights": f"weights/{r.checkpoint_id}", "sha256": sha256(dst), "projection": mid.endswith("+proj"),
                                           "training_pool": r.training_pool_id, "seed": int(r.seed), "selection": r.get("select_crit"), "params": int(r.params),
                                           "train_updates": info.get("iters", 64000), "best_val_psnr_it": info.get("best_psnr_it"), "best_val_ssim_it": info.get("best_ssim_it"),
                                           "n_pool_images": info.get("n_pool_images"), "n_units": info.get("n_units")}
    (SUB / "model_manifest.json").write_text(json.dumps(man, indent=1))
    history(SUB / "results" / "history.json")
    for f in ("role_table.csv", "method_summary.csv", "paired.json", "paired.csv", "per_image.csv", "residuals.csv", "seeds.csv", "fig_tradeoff_val.png", "fig_tradeoff_urban.png", "roles_frozen.json", "latency.json", "policy_lock.json"):
        shutil.copy2(RES / f, SUB / "results" / f)


def history(path: Path) -> None:
    """Earlier completed measurements quoted in the report, re-read from their record files (not from chat summaries)."""
    r = MD / "results"
    pilot = MD.parent / "pilot_outputs"
    f = pd.read_json(r / "M2f.jsonl", lines=True)
    f = f[f.consistency_at_output & ~f.tta]
    lam = {str(l): {sp: round(float(g[g.split == sp].psnr.mean()), 3) for sp in ("val", "fresh", "assignment", "div2k_eval20")} for l, g in f.groupby("lambda")}
    srd = {k: json.loads((r / f"srd_{k}_final_bal_s20260915_info.json").read_text()) for k in "ABC"}
    b10 = pd.read_json(pilot / "B10_oblique.jsonl", lines=True)
    fin = pd.read_json(r / "FINAL.jsonl", lines=True)
    fin = fin[fin.consistency_at_output & ~fin.tta]
    m1 = pd.read_json(r / "M1_consistency.jsonl", lines=True)
    g = pd.read_json(r / "M2g.jsonl", lines=True)
    g = g[g.consistency_at_output & ~g.tta]
    e = pd.read_json(r / "M2e.jsonl", lines=True)
    e = e[e.consistency_at_output & ~e.tta]
    out = {
        "lambda_experiment_psnr": lam,
        "srd": {k: {"params": v["params"], "val_start": v["val_start"], "val_after_fit": v.get("val_after_fit"), "best_val": v["best_val"]} for k, v in srd.items()},
        "b10_bound_holds": f"{int(b10.bound_holds.sum())}/{len(b10)}",
        "b10_best_linear_consistent_8x": b10[b10.scale == 8].groupby("upscaler").psnr_LS_projection.mean().round(3).to_dict(),
        "previous_core_8x": fin.groupby(["seed", "split"]).agg(psnr=("psnr", "mean"), ssim=("ssim", "mean")).round(4).reset_index().to_dict(orient="records"),
        "m1_rows": len(m1),
        "m2g_sampling_val": g[g.split == "val"].groupby("variant").psnr.mean().round(3).to_dict(),
        "m2e_data_val": e[e.split == "val"].groupby("variant").psnr.mean().round(3).to_dict(),
        "pool_decision": json.loads((r / "final_scales" / "pool_decision.json").read_text()),
        "final_scale_runs": {p.stem: {k: v for k, v in json.loads(p.read_text()).items() if k not in ("curve", "pool_keys")} for p in sorted((r / "final_scales").glob("x*_s*.json"))},
    }
    path.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())

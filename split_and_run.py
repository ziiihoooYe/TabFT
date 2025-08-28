import argparse, copy, os, re, subprocess, sys, yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ---------- util ----------

# --- Constants for result parsing (inspired by your parsing script) ---
RE_METRIC = re.compile(r"([A-Za-z0-9_]+)\s+MEAN\s+=\s+([-+0-9eE.]+)")
CLASS_METRICS = {"Accuracy", "Avg_Recall", "F1", "AUC"}
REG_METRICS   = {"MAE", "R2", "RMSE"}

def has_results(log_path: Path) -> bool:
    """
    Checks if a log file exists and contains a complete set of either
    classification or regression metrics, indicating a successful run.
    """
    if not log_path.is_file():
        return False

    found_metrics: Set[str] = set()
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                match = RE_METRIC.search(line)
                if match:
                    found_metrics.add(match.group(1))
    except (IOError, UnicodeDecodeError):
        # If the file is unreadable or malformed, treat it as not having results.
        return False

    # A run is complete if it has all metrics for either task type.
    is_classification_complete = CLASS_METRICS.issubset(found_metrics)
    is_regression_complete = REG_METRICS.issubset(found_metrics)

    return is_classification_complete or is_regression_complete

def even_chunks(seq: List[Any], k: int) -> List[List[Any]]:
    k = max(1, min(k, len(seq))); q, r = divmod(len(seq), k)
    sizes = [q + (i < r) for i in range(k)]
    out, idx = [], 0
    for sz in sizes: out.append(seq[idx:idx+sz]); idx += sz
    return [c for c in out if c]

def pair_prod(a, b): return [(x, y) for x in a for y in b]

def save_yaml(cfg, p: Path): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(yaml.safe_dump(cfg, sort_keys=False))

def build_env(gpu: int | None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return env

def run_cmd(cmd: List[str], env):                  # ← Directly print for debugging
    print("⇢", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, check=False).returncode

# ---------- top-level worker (picklable) ----------
def worker(job: Tuple[Path, Path, int | None]) -> Tuple[str, int]:
    yaml_p, log_p, gpu = job

    log_p.parent.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(yaml_p.read_text())
    cfg["gpu"] = gpu if gpu is not None else cfg.get("gpu", 0)
    yaml_p.write_text(yaml.safe_dump(cfg, sort_keys=False))

    cmd = [sys.executable, "main.py",
           "--config", str(yaml_p),
           "--log",    str(log_p)]
    rc = subprocess.run(cmd,
                        env=build_env(gpu),
                        check=False).returncode
    return yaml_p.name, rc

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, type=Path)
    ap.add_argument("--logdir", required=True, type=Path)
    ap.add_argument("--mode", choices=["dataset", "model", "combo"], default="dataset")
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--gpus", type=int, nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = yaml.safe_load(args.yaml.read_text())
    jobs: list[Tuple[Path, Path, int | None]] = []

    if args.mode == "dataset":
        for i, dchunk in enumerate(even_chunks(base["dataset"], args.n_splits)):
            cfg = copy.deepcopy(base); cfg["dataset"] = dchunk
            yml = args.yaml.with_stem(f"{args.yaml.stem}_ds{i}"); save_yaml(cfg, yml)
            log = args.logdir / f"{yml.stem}.log"

            # --- Check if valid results already exist in the log ---
            if has_results(log):
                print(f"Skipping '{yml.stem}': Valid results found in log.")
                continue

            gpu = args.gpus[i % len(args.gpus)] if args.gpus else None
            jobs.append((yml, log, gpu))

    elif args.mode == "model":
        for i, mchunk in enumerate(even_chunks(base["model_type"], args.n_splits)):
            cfg = copy.deepcopy(base); cfg["model_type"] = mchunk
            yml = args.yaml.with_stem(f"{args.yaml.stem}_mdl{i}"); save_yaml(cfg, yml)
            log = args.logdir / f"{yml.stem}.log"

            # --- Check if valid results already exist in the log ---
            if has_results(log):
                print(f"Skipping '{yml.stem}': Valid results found in log.")
                continue

            gpu = args.gpus[i % len(args.gpus)] if args.gpus else None
            jobs.append((yml, log, gpu))

    else:  # combo
        for j, (ds, mdl) in enumerate(pair_prod(base["dataset"], base["model_type"])):
            cfg = copy.deepcopy(base); cfg["dataset"], cfg["model_type"] = [ds], [mdl]
            yml = args.yaml.with_stem(f"{args.yaml.stem}_{ds}__{mdl}"); save_yaml(cfg, yml)
            log = args.logdir / ds / f"{mdl}.log"

            # --- Check if valid results already exist in the log ---
            if has_results(log):
                print(f"Skipping '{yml.stem}': Valid results found in log.")
                continue

            gpu = args.gpus[j % len(args.gpus)] if args.gpus else None
            jobs.append((yml, log, gpu))

    print(f"Prepared {len(jobs)} new job(s) to run.")
    if args.dry_run: return

    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        for fut in as_completed(pool.submit(worker, j) for j in jobs):
            name, rc = fut.result(); print(f"[{'✓' if rc==0 else f'✗({rc})'}] {name}", flush=True)

if __name__ == "__main__":
    main()
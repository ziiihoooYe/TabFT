import argparse, copy, re, subprocess, yaml
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from datetime import datetime

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

def save_yaml(cfg, p: Path): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _choose_split_axis(base: Dict[str, Any]) -> Tuple[str | None, List[Any]]:
    """
    Choose a coarse split axis:
      - Prefer 'dataset' if it's a list.
      - Else use 'model_type' if it's a list.
      - Else return (None, [None]) meaning no natural axis; we'll replicate the config.
    """
    ds = base.get("dataset", None)
    mt = base.get("model_type", None)
    if isinstance(ds, list) and len(ds) > 0:
        return "dataset", ds
    if isinstance(mt, list) and len(mt) > 0:
        return "model_type", mt
    return None, [None]


def _prepare_simple_chunks(base: Dict[str, Any],
                           yaml_path: Path,
                           logdir: Path,
                           num_chunks: int) -> List[Tuple[Path, Path]]:
    """
    Coarse chunking: split only along one selected axis into `num_chunks` pieces.
    This does NOT create the fine-grained (dataset, model_type) product.
    Returns a list of (chunk_yaml_path, log_path) tuples, skipping any whose log already has complete results.
    """
    axis, items = _choose_split_axis(base)
    # If no natural axis, just replicate the config into `num_chunks` copies.
    chunks = even_chunks(items, num_chunks) if axis is not None else [[None] for _ in range(num_chunks)]

    jobs: list[Tuple[Path, Path]] = []
    for i, ch in enumerate(chunks):
        cfg = copy.deepcopy(base)
        if axis == "dataset":
            cfg["dataset"] = ch
        elif axis == "model_type":
            cfg["model_type"] = ch
        # else: replicate unchanged

        yml = yaml_path.with_stem(f"{yaml_path.stem}_chunk{i:02d}")
        save_yaml(cfg, yml)
        yml = yml.resolve()

        log = logdir / f"{yml.stem}.log"
        log = log.resolve()
        if has_results(log):
            print(f"Skipping '{yml.stem}': Valid results found in log.")
            continue
        jobs.append((yml, log))
    return jobs


def _write_sbatch_simple_array(jobs: List[Tuple[Path, Path]],
                               *,
                               working_dir: Path,
                               job_name: str,
                               cpus: int,
                               gpus_per_task: int,
                               mem: str,
                               partition: str,
                               time_limit: str,
                               conda_env: str | None,
                               max_parallel: int) -> Path:
    """
    Create a self-contained SBATCH script that runs one chunk per array task.
    Each array task selects its YAML and LOG path from bash arrays.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = working_dir / f"{job_name}_{ts}"
    logs_root = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    script_path = run_root / "slurm_launch.sh"

    # Prepare bash arrays for YAMLs and LOGs
    ymls = [str(y) for (y, _) in jobs]
    logs = [str(l) for (_, l) in jobs]
    yml_list = " ".join(f'"{p}"' for p in ymls)
    log_list = " ".join(f'"{p}"' for p in logs)

    array_spec = f"0-{len(jobs)-1}"
    if max_parallel and max_parallel > 0:
        array_spec = f"{array_spec}%{max_parallel}"

    sbatch_lines = []
    sbatch_lines.append("#!/bin/bash")
    sbatch_lines.append(f"#SBATCH --job-name={job_name}")
    sbatch_lines.append(f"#SBATCH --array={array_spec}")
    sbatch_lines.append(f"#SBATCH --time={time_limit}")
    sbatch_lines.append(f"#SBATCH --cpus-per-task={cpus}")
    sbatch_lines.append(f"#SBATCH --mem={mem}")
    sbatch_lines.append(f"#SBATCH --partition={partition}")
    sbatch_lines.append("#SBATCH --ntasks=1")
    sbatch_lines.append("#SBATCH --nodes=1")
    if gpus_per_task and gpus_per_task > 0:
        sbatch_lines.append(f"#SBATCH --gres=gpu:{gpus_per_task}")
    # Always capture a minimal slurm wrapper log; main logs go to LOGS array.
    sbatch_lines.append(f"#SBATCH --output={logs_root}/slurm_%A_%a.out")
    sbatch_lines.append(f"#SBATCH --error={logs_root}/slurm_%A_%a.err")
    sbatch_lines.append("")
    sbatch_lines.append("set -euo pipefail")
    sbatch_lines.append("")
    sbatch_lines.append('cd "$SLURM_SUBMIT_DIR"')
    sbatch_lines.append("")
    sbatch_lines.append(f'YAMLS=({yml_list})')
    sbatch_lines.append(f'LOGS=({log_list})')
    sbatch_lines.append('IDX="${SLURM_ARRAY_TASK_ID}"')
    sbatch_lines.append('YAML="${YAMLS[$IDX]}"')
    sbatch_lines.append('LOG="${LOGS[$IDX]}"')
    sbatch_lines.append('mkdir -p "$(dirname "$LOG")"')
    sbatch_lines.append("")
    # env activation (conda preferred)
    if conda_env:
        sbatch_lines.append('source "$(conda info --base)/etc/profile.d/conda.sh"')
        sbatch_lines.append(f"conda activate {conda_env}")
    sbatch_lines.append('echo "Using Python: $(which python)"')
    sbatch_lines.append('echo "Python version: $(python -V)"')
    sbatch_lines.append("")
    sbatch_lines.append('echo "App log (from program): $LOG"')
    sbatch_lines.append('echo "Stdout redirected to: ${LOG%.log}.stdout"')
    sbatch_lines.append('echo "Stderr redirected to: ${LOG%.log}.stderr"')
    sbatch_lines.append('python -u main.py --config "$YAML" --log "$LOG" > "${LOG%.log}.stdout" 2> "${LOG%.log}.stderr"')
    sbatch_lines.append("")
    script_path.write_text("\n".join(sbatch_lines), encoding="utf-8")
    return script_path


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, type=Path, help="Path to the base YAML config.")
    ap.add_argument("--logdir", required=True, type=Path, help="Directory where per-chunk logs will be written.")
    ap.add_argument("--num-chunks", type=int, default=8, help="Number of coarse chunks (default: 8).")
    ap.add_argument("--sbatch-working-dir", type=Path, default=Path("sbatch"), help="Where to place generated SBATCH run folders.")
    ap.add_argument("--job-name", type=str, default=None, help="Slurm job name (default: <yaml_stem>_chunks).")
    ap.add_argument("--cpus", type=int, default=8, help="CPUs per task.")
    ap.add_argument("--gpus-per-task", type=int, default=0, help="GPUs per task (Slurm level).")
    ap.add_argument("--mem", type=str, default="32G", help="Memory per task, e.g., 32G.")
    ap.add_argument("--partition", type=str, default="general", help="Slurm partition name.")
    ap.add_argument("--time", type=str, default="2-00:00:00", help="Time limit, e.g., 1-00:00:00.")
    ap.add_argument("--max-parallel", type=int, default=8, help="Max concurrent array tasks, e.g., 8 for clusters that cap per-user concurrency.")
    ap.add_argument("--conda-env", type=str, default=None, help="Conda env to activate inside the job.")
    ap.add_argument("--dry-run", action="store_true", help="Only generate the SBATCH script; do not submit.")
    args = ap.parse_args()

    base = yaml.safe_load(args.yaml.read_text())
    jobs = _prepare_simple_chunks(base, args.yaml, args.logdir, args.num_chunks)

    print(f"Prepared {len(jobs)} chunk(s).")
    if len(jobs) == 0:
        print("Nothing to do: all chunks already have complete results in logs.")
        return

    job_name = args.job_name or f"{args.yaml.stem}_chunks"
    script = _write_sbatch_simple_array(
        jobs,
        working_dir=args.sbatch_working_dir,
        job_name=job_name,
        cpus=args.cpus,
        gpus_per_task=args.gpus_per_task,
        mem=args.mem,
        partition=args.partition,
        time_limit=args.time,
        conda_env=args.conda_env,
        max_parallel=args.max_parallel,
    )
    print(f"SBATCH script written to: {script}")
    if args.dry_run:
        print("[DRY-RUN] Not submitting to Slurm.")
        return

    out = subprocess.run(["sbatch", str(script)], check=False, capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())

if __name__ == "__main__":
    main()
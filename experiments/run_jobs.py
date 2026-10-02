"""Run a list of jobs across GPUs, one job per GPU at a time.

Each line of the job file is `name<TAB>command`. A job is skipped when runs/.done/<name>
exists, so the file can be extended and rerun. Output goes to ../logs/<name>.log.
"""

import argparse
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("jobs")
ap.add_argument("--gpus", nargs="+", required=True)
ap.add_argument("--logs", default="../logs")
ap.add_argument("--per-gpu", type=int, default=1)
args = ap.parse_args()

done_dir, claim_dir = Path("runs/.done"), Path("runs/.claim")
done_dir.mkdir(parents=True, exist_ok=True)
claim_dir.mkdir(parents=True, exist_ok=True)
Path(args.logs).mkdir(parents=True, exist_ok=True)
todo: queue.Queue = queue.Queue()
for line in Path(args.jobs).read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        name, _, cmd = line.partition("\t")
        if not (done_dir / name).exists():
            todo.put((name, cmd))
print(f"{todo.qsize()} jobs to run", flush=True)


def worker(gpu: str) -> None:
    while True:
        try:
            name, cmd = todo.get_nowait()
        except queue.Empty:
            return
        if (done_dir / name).exists():
            continue
        # Several runners can share one job file: a job is claimed by creating its lock file,
        # which fails if another runner got there first.
        try:
            os.close(os.open(claim_dir / name, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        t0 = time.time()
        # Cap math-library threads: every job also runs a pool of descriptor workers, and
        # uncapped they oversubscribe a shared machine many times over.
        threads = {k: "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=".", PYTHONUNBUFFERED="1", **threads)
        with open(Path(args.logs) / f"{name}.log", "w") as log:
            code = subprocess.call(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        if code == 0:
            (done_dir / name).touch()
        (claim_dir / name).unlink(missing_ok=True)
        print(f"{'ok  ' if code == 0 else 'FAIL'} {name} gpu{gpu} {time.time() - t0:.0f}s", flush=True)


threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus for _ in range(args.per_gpu)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("queue finished", flush=True)

#!/bin/bash -l
# Run any `python run.py ...` command as a CPU-only Slurm job on Euler. The methods only call the
# CSCS inference API, so no GPU is needed — but Euler compute nodes have no direct internet access.
# This script routes the calls through the ETH web proxy and aborts early if the API is still
# unreachable; in that case run the same command on a login node inside tmux (see README.md).
# Usage, from track_2a/:
#   sbatch euler/api_job.sh run --method m2 --manifest dev60 --workers 4
#   sbatch euler/api_job.sh check
#SBATCH --job-name=uzhdiff
#SBATCH --account=es_chatzi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=uzhdiff-%j.out
set -u
export PYTHONUNBUFFERED=1
export http_proxy=http://proxy.ethz.ch:3128 https_proxy=http://proxy.ethz.ch:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
cd "$(dirname "$(readlink -f "$0")")/.."
echo "host: $(hostname)  python: $(python3 --version 2>&1)  started: $(date)"

# 20-second reachability test before spending the job's time budget.
python3 - <<'EOF' || { echo "计算节点连不上接口（经 proxy.ethz.ch 也不行）。请在登录节点用 tmux 跑，见 euler/README.md"; exit 1; }
import os, sys, urllib.request
sys.path.insert(0, "src")
from uzh_diff.config import get_llm_config
cfg = get_llm_config(interactive=False)
req = urllib.request.Request(cfg.base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer " + cfg.api_key})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        print("api reachable, HTTP", r.status)
except Exception as e:  # noqa: BLE001
    sys.exit("api unreachable from %s: %s" % (os.uname().nodename, e))
EOF

python3 run.py "$@"
echo "finished: $(date)"

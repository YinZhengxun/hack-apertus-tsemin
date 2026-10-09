"""Launcher: `python run.py <command>` from the track_2a directory. Needs Python 3.7+ and nothing else."""
import sys
from pathlib import Path

if sys.version_info < (3, 7):
    sys.exit("需要 Python 3.7 或更高版本，当前是 %d.%d" % sys.version_info[:2])

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uzh_diff.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

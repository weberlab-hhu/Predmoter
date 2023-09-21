import subprocess
import os

PREDMOTER_VERSION = "0.3.2"
GIT_COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=os.path.dirname(os.path.realpath(__file__))).decode("ascii").strip()
EPS = 1e-8
MAX_VALUES_IN_RAM = 21_384_000

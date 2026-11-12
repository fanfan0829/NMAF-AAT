from pathlib import Path
import shutil
from allen_cahn_fixed_activation_seed99_common import RESULT_ROOT, SEEDS, training_points_path, generate_points_for_seed, train_single_seed
import numpy as np

SEED = 99

if __name__ == "__main__":
    if SEEDS != [99]:
        raise RuntimeError(f"Expected SEEDS=[99], got {SEEDS}")
    point_path = training_points_path(SEED)
    point_path.parent.mkdir(parents=True, exist_ok=True)
    if not point_path.exists():
        np.savez_compressed(point_path, **generate_points_for_seed(SEED))
    stale = RESULT_ROOT / "mish" / f"seed_{SEED}"
    if stale.exists():
        shutil.rmtree(stale)
        print(f"Removed stale Mish output: {stale}")
    train_single_seed("mish", SEED)

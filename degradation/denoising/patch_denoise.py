import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from concurrent.futures import ProcessPoolExecutor
import time

# =========================================================
# CONFIG
# =========================================================

INPUT_PARQUET = "/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/patch_n_split/patch_metadata.parquet"

OUTPUT_BASE = Path("/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/patch_noisy")

BASE_REL = Path("/Data5/data5_cvpr_pg_2/dataset/patch_n_split")

NOISE_LEVELS = [10, 30, 50, 70, 90]

NUM_WORKERS = 8
SAVE_EVERY = 200
SKIP_EXISTING = False

# =========================================================
# ADD GAUSSIAN NOISE (same logic as your original code)
# =========================================================

def add_gaussian_noise(cube: np.ndarray, sigma: int):

    cube = cube.astype(np.float32)

    data_max = cube.max()

    if data_max <= 0:
        return cube.copy()

    # Normalize to [0, 1]
    cube_scaled = cube / (data_max + 1e-8)

    # Gaussian noise
    noise = np.random.normal(
        loc=0.0,
        scale=sigma / 255.0,
        size=cube_scaled.shape
    ).astype(np.float32)

    noisy = cube_scaled + noise

    # Clip
    noisy = np.clip(noisy, 0.0, 1.0)

    # Scale back
    noisy = noisy * data_max

    return noisy.astype(np.float32)

# =========================================================
# PROCESS ONE PATCH FOR ALL NOISE LEVELS
# =========================================================

def process_patch(row):

    results = []

    try:
        patch_path = Path(row["patch_path"])

        rel = patch_path.relative_to(BASE_REL)

        with rasterio.open(patch_path) as src:
            cube = src.read().astype(np.float32)
            profile = src.profile.copy()

        profile.update(dtype="float32")

        stem = patch_path.stem

        for sigma in NOISE_LEVELS:

            noise_name = f"{stem}_{sigma}.tif"

            out_path = OUTPUT_BASE / f"noise_{sigma}" / rel.parent / noise_name

            out_path.parent.mkdir(parents=True, exist_ok=True)

            if SKIP_EXISTING and out_path.exists():

                results.append({
                    "patch_id": row["patch_id"],
                    "patch_name": row["patch_name"],
                    "noise_name": noise_name,
                    "scene_name": row["scene_name"],
                    "country": row["country"],
                    "section": row["section"],
                    "noise_level": sigma,
                    "patch_path": row["patch_path"],
                    "noise_path": str(out_path)
                })

                continue

            noisy_cube = add_gaussian_noise(cube, sigma)

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(noisy_cube)

            results.append({
                "patch_id": row["patch_id"],
                "patch_name": row["patch_name"],
                "noise_name": noise_name,
                "scene_name": row["scene_name"],
                "country": row["country"],
                "section": row["section"],
                "noise_level": sigma,
                "patch_path": row["patch_path"],
                "noise_path": str(out_path)
            })

        return {"ok": True, "rows": results}

    except Exception as e:

        return {
            "ok": False,
            "patch_id": row.get("patch_id"),
            "patch_path": row.get("patch_path"),
            "error": str(e)
        }

# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata: {INPUT_PARQUET}")

    df = pd.read_parquet(INPUT_PARQUET)

    total = len(df)

    print(f"Total clean patches : {total:,}")
    print(f"Noise levels        : {NOISE_LEVELS}")
    print(f"Total noisy patches : {total * len(NOISE_LEVELS):,}")
    print(f"CPU workers         : {NUM_WORKERS}\n")

    rows = df.to_dict(orient="records")

    metadata = []
    failed = []

    start = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:

        for i, res in enumerate(ex.map(process_patch, rows, chunksize=4), start=1):

            if res["ok"]:
                metadata.extend(res["rows"])
            else:
                failed.append(res)

            # periodic save
            if i % SAVE_EVERY == 0:

                md_df = pd.DataFrame(metadata)

                for sigma in NOISE_LEVELS:

                    out_parquet = OUTPUT_BASE / f"noise_{sigma}" / f"patch_metadata_noise_{sigma}.parquet"

                    md_df[md_df["noise_level"] == sigma].to_parquet(
                        out_parquet,
                        index=False
                    )

            # progress
            if i % 10 == 0 or i == total:

                elapsed = time.time() - start
                speed = i / elapsed if elapsed > 0 else 0.0

                eta = (total - i) / speed if speed > 0 else 0

                print(
                    f"[{i:6d}/{total}] "
                    f"{100*i/total:6.2f}% | "
                    f"speed={speed:.2f} patch/s | "
                    f"elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                    f"eta={time.strftime('%H:%M:%S', time.gmtime(eta))}",
                    flush=True
                )

    # final save
    md_df = pd.DataFrame(metadata)

    for sigma in NOISE_LEVELS:

        out_parquet = OUTPUT_BASE / f"noise_{sigma}" / f"patch_metadata_noise_{sigma}.parquet"

        md_df[md_df["noise_level"] == sigma].to_parquet(
            out_parquet,
            index=False
        )

    if failed:

        pd.DataFrame(failed).to_csv(
            OUTPUT_BASE / "patch_noisy_failed.csv",
            index=False
        )

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Clean patches       : {total:,}")
    print(f"Noisy patches       : {len(metadata):,}")
    print(f"Failed patches      : {len(failed):,}")
    print(f"Elapsed             : {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

if __name__ == "__main__":
    main()
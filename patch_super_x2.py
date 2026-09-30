import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
import scipy.io as sio
from skimage.transform import resize

# =========================================================
# CONFIG
# =========================================================

INPUT_PARQUET = "/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/patch_n_split/patch_metadata.parquet"

MNF_PARQUET = "/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/patch_split_mnf/patch_metadata_mnf_all.parquet"

OUTPUT_BASE = Path("/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/patch_super/x2")

OUTPUT_PARQUET = OUTPUT_BASE / "patch_metadata_x2.parquet"
FAILED_CSV = OUTPUT_BASE / "patch_metadata_x2.failed.csv"

SCALE = 2

NUM_WORKERS = 8
SAVE_EVERY = 200
SKIP_EXISTING = False

# =========================================================
# CREATE ONE SR SAMPLE
# =========================================================

def process_row(row):

    try:
        patch_path = Path(row["patch_path"])
        mnf_path = Path(row["mnf_path"])

        section = row["section"]
        country = row["country"]

        # preserve exact original scene folder
        scene_folder = patch_path.parent.name

        out_dir = OUTPUT_BASE / section / country / scene_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = patch_path.stem
        sr_name = f"{base_name}_X{SCALE}.mat"
        sr_path = out_dir / sr_name

        if SKIP_EXISTING and sr_path.exists():

            return {
                "ok": True,
                "row": {
                    "patch_id": row["patch_id"],
                    "patch_name": row["patch_name"],
                    "scene_name": row["scene_name"],
                    "country": row["country"],
                    "section": row["section"],
                    "patch_path": row["patch_path"],
                    "mnf_name": row["mnf_name"],
                    "mnf_path": row["mnf_path"],
                    "scale": SCALE,
                    "sr_name": sr_name,
                    "sr_path": str(sr_path)
                }
            }

        # -------------------------------------------------
        # Read GT (MNF)
        # -------------------------------------------------
        with rasterio.open(mnf_path) as src:
            gt = src.read().astype(np.float32)   # (B,H,W)

        gt = np.transpose(gt, (1, 2, 0))         # (H,W,B)

        # -------------------------------------------------
        # Read original input
        # -------------------------------------------------
        with rasterio.open(patch_path) as src:
            inp = src.read().astype(np.float32)

        inp = np.transpose(inp, (1, 2, 0))       # (H,W,B)

        # -------------------------------------------------
        # Normalize to [0,1]
        # -------------------------------------------------
        inp = inp - inp.min()
        if inp.max() > 0:
            inp = inp / inp.max()

        gt = gt - gt.min()
        if gt.max() > 0:
            gt = gt / gt.max()

        H, W, B = inp.shape

        # -------------------------------------------------
        # Create LR (X2)
        # -------------------------------------------------
        lr = resize(
            inp,
            (H // SCALE, W // SCALE, B),
            order=3,
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.float32)

        # -------------------------------------------------
        # Bicubic upsample
        # -------------------------------------------------
        bicubic = resize(
            lr,
            (H, W, B),
            order=3,
            anti_aliasing=False,
            preserve_range=True
        ).astype(np.float32)

        # -------------------------------------------------
        # Save MAT
        # -------------------------------------------------
        sio.savemat(
            sr_path,
            {
                "gt": gt.astype(np.float32),
                "ms": lr.astype(np.float32),
                "ms_bicubic": bicubic.astype(np.float32)
            }
        )

        return {
            "ok": True,
            "row": {
                "patch_id": row["patch_id"],
                "patch_name": row["patch_name"],
                "scene_name": row["scene_name"],
                "country": row["country"],
                "section": row["section"],
                "patch_path": row["patch_path"],
                "mnf_name": row["mnf_name"],
                "mnf_path": row["mnf_path"],
                "scale": SCALE,
                "sr_name": sr_name,
                "sr_path": str(sr_path)
            }
        }

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

    print(f"Input metadata : {INPUT_PARQUET}")
    print(f"MNF metadata   : {MNF_PARQUET}")

    df_input = pd.read_parquet(INPUT_PARQUET)
    df_mnf = pd.read_parquet(MNF_PARQUET)

    # merge only MNF-specific columns
    df = df_input.merge(
        df_mnf[["patch_id", "mnf_name", "mnf_path"]],
        on="patch_id",
        how="inner"
    )

    total = len(df)

    print(f"Scale          : X{SCALE}")
    print(f"Total patches  : {total:,}")
    print(f"CPU workers    : {NUM_WORKERS}\n")

    rows = df.to_dict(orient="records")

    metadata = []
    failed = []

    start = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:

        futures = [ex.submit(process_row, r) for r in rows]

        for i, fut in enumerate(as_completed(futures), start=1):

            res = fut.result()

            if res["ok"]:
                metadata.append(res["row"])
            else:
                failed.append(res)

            # periodic save
            if i % SAVE_EVERY == 0:
                pd.DataFrame(metadata).to_parquet(OUTPUT_PARQUET, index=False)

            # live progress
            if i % 10 == 0 or i == total:

                elapsed = time.time() - start
                speed = i / elapsed if elapsed > 0 else 0.0
                eta = (total - i) / speed if speed > 0 else 0

                print(
                    f"\r[{i:6d}/{total}] "
                    f"{100*i/total:6.2f}% | "
                    f"mat={i} | "
                    f"speed={speed:.2f} patch/s | "
                    f"elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                    f"eta={time.strftime('%H:%M:%S', time.gmtime(eta))}",
                    end="",
                    flush=True
                )

    print()

    # final save
    pd.DataFrame(metadata).to_parquet(OUTPUT_PARQUET, index=False)

    if failed:
        pd.DataFrame(failed).to_csv(FAILED_CSV, index=False)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("DONE X2")
    print("=" * 60)
    print(f"Patches created : {len(metadata):,}")
    print(f"Failed          : {len(failed):,}")
    print(f"Elapsed         : {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"Metadata        : {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
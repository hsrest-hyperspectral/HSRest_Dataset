#!/usr/bin/env python3

import os
import shutil
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = Path(
    "/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/patch_n_split"
)

OUTPUT_ROOT = Path(
    "/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/inpaint_data"
)

SEED = 2026

TARGET_COUNTS = {
    "train": 70,
    "val": 10,
    "test": 20,
}

# 1-based HSI band numbers
RED_BAND = 42
GREEN_BAND = 22
BLUE_BAND = 11


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# PROGRESS DISPLAY
# ============================================================

def format_time(seconds):
    """Convert seconds to HH:MM:SS."""

    if seconds < 0 or not np.isfinite(seconds):
        return "--:--:--"

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(current, total, start_time, filename=""):
    """
    Print live progress with:
        percentage
        current / total
        files/sec
        elapsed
        ETA
        current file
    """

    elapsed = time.time() - start_time

    speed = current / elapsed if elapsed > 0 else 0

    remaining = total - current

    eta = (
        remaining / speed
        if speed > 0
        else float("inf")
    )

    percent = (
        100.0 * current / total
        if total > 0
        else 0
    )

    # Keep filename from making terminal line enormous
    max_filename_length = 55

    if len(filename) > max_filename_length:
        filename = "..." + filename[-(max_filename_length - 3):]

    message = (
        f"\r"
        f"[{percent:6.2f}%] "
        f"{current:6d}/{total:<6d} | "
        f"{speed:6.2f} files/s | "
        f"Elapsed: {format_time(elapsed)} | "
        f"ETA: {format_time(eta)} | "
        f"Current: {filename:<55}"
    )

    print(
        message,
        end="",
        flush=True
    )

    if current == total:
        print()


# ============================================================
# PATH INFORMATION
# ============================================================

def get_scene_name(tif_path):

    scene_dir = tif_path.parent
    region_dir = scene_dir.parent

    scene = scene_dir.name
    region = region_dir.name

    return region, scene


# ============================================================
# PATCH STATISTICS
# ============================================================

def calculate_patch_statistics(tif_path):

    cube = tifffile.imread(str(tif_path))

    if cube.ndim != 3:
        raise ValueError(
            f"Unexpected TIFF dimensions "
            f"{cube.shape}: {tif_path}"
        )

    # Expected H x W x B
    if cube.shape[-1] == 175:

        hsi = cube

    # Possible B x H x W
    elif cube.shape[0] == 175:

        hsi = np.transpose(
            cube,
            (1, 2, 0)
        )

    else:

        raise ValueError(
            f"Cannot identify spectral dimension "
            f"for shape {cube.shape}: {tif_path}"
        )

    hsi = hsi.astype(
        np.float32,
        copy=False
    )

    # --------------------------------------------------------
    # Overall intensity
    # --------------------------------------------------------

    mean_intensity = float(
        np.nanmean(hsi)
    )

    intensity_std = float(
        np.nanstd(hsi)
    )

    # --------------------------------------------------------
    # Spectral variation
    # --------------------------------------------------------

    pixel_spectral_std = np.nanstd(
        hsi,
        axis=2
    )

    spectral_variation = float(
        np.nanmean(pixel_spectral_std)
    )

    # --------------------------------------------------------
    # Spatial variation
    # --------------------------------------------------------

    mean_image = np.nanmean(
        hsi,
        axis=2
    )

    spatial_variation = float(
        np.nanstd(mean_image)
    )

    return {
        "mean_intensity": mean_intensity,
        "intensity_std": intensity_std,
        "spectral_variation": spectral_variation,
        "spatial_variation": spatial_variation,
    }


# ============================================================
# PROPORTIONAL ALLOCATION
# ============================================================

def allocate_proportional(total_samples, counts):

    counts = np.asarray(
        counts,
        dtype=float
    )

    if counts.sum() == 0:
        return np.zeros(
            len(counts),
            dtype=int
        )

    raw = (
        total_samples
        * counts
        / counts.sum()
    )

    allocation = np.floor(
        raw
    ).astype(int)

    remaining = (
        total_samples
        - allocation.sum()
    )

    if remaining > 0:

        remainders = raw - allocation

        order = np.argsort(
            -remainders
        )

        for i in order[:remaining]:
            allocation[i] += 1

    return allocation


# ============================================================
# QUANTILE STRATA
# ============================================================

def assign_quantile_strata(df):

    df = df.copy()

    df["brightness_bin"] = 0
    df["spectral_bin"] = 0
    df["spatial_bin"] = 0

    for scene_name, indices in df.groupby("scene").groups.items():

        scene_df = df.loc[indices]

        # Rank first handles duplicate values safely
        df.loc[indices, "brightness_bin"] = pd.qcut(
            scene_df["mean_intensity"].rank(
                method="first"
            ),
            q=3,
            labels=False
        )

        df.loc[indices, "spectral_bin"] = pd.qcut(
            scene_df["spectral_variation"].rank(
                method="first"
            ),
            q=3,
            labels=False
        )

        df.loc[indices, "spatial_bin"] = pd.qcut(
            scene_df["spatial_variation"].rank(
                method="first"
            ),
            q=3,
            labels=False
        )

    df["stratum"] = (
        df["brightness_bin"].astype(str)
        + "_"
        + df["spectral_bin"].astype(str)
        + "_"
        + df["spatial_bin"].astype(str)
    )

    return df


# ============================================================
# SELECT WITHIN SCENE
# ============================================================

def select_from_scene(scene_df, n_samples, rng):

    if n_samples >= len(scene_df):
        return scene_df.index.tolist()

    scene_df = assign_quantile_strata(
        scene_df
    )

    strata = list(
        scene_df.groupby("stratum")
    )

    selected = []

    shuffled_strata = strata.copy()

    rng.shuffle(
        shuffled_strata
    )

    # If fewer samples than strata,
    # give different strata one sample each.
    if n_samples < len(shuffled_strata):

        chosen_strata = (
            shuffled_strata[:n_samples]
        )

        for _, group in chosen_strata:

            idx = rng.choice(
                group.index.to_numpy()
            )

            selected.append(idx)

        return selected

    # --------------------------------------------------------
    # Proportional allocation among strata
    # --------------------------------------------------------

    stratum_counts = [
        len(group)
        for _, group in strata
    ]

    allocation = allocate_proportional(
        n_samples,
        stratum_counts
    )

    for (
        (stratum_name, group),
        quota
    ) in zip(
        strata,
        allocation
    ):

        if quota <= 0:
            continue

        quota = min(
            quota,
            len(group)
        )

        chosen = rng.choice(
            group.index.to_numpy(),
            size=quota,
            replace=False
        )

        selected.extend(
            chosen.tolist()
        )

    # Safety
    if len(selected) < n_samples:

        remaining = list(
            set(scene_df.index)
            - set(selected)
        )

        rng.shuffle(
            remaining
        )

        selected.extend(
            remaining[:n_samples - len(selected)]
        )

    elif len(selected) > n_samples:

        rng.shuffle(
            selected
        )

        selected = selected[:n_samples]

    return selected


# ============================================================
# FCC PNG
# ============================================================

def make_fcc_png(tif_path, output_png):

    cube = tifffile.imread(
        str(tif_path)
    )

    if cube.ndim != 3:
        raise ValueError(
            f"Unexpected TIFF shape "
            f"{cube.shape}: {tif_path}"
        )

    if cube.shape[-1] == 175:

        hsi = cube

    elif cube.shape[0] == 175:

        hsi = np.transpose(
            cube,
            (1, 2, 0)
        )

    else:

        raise ValueError(
            f"Cannot determine band axis: "
            f"{cube.shape}"
        )

    hsi = hsi.astype(
        np.float32,
        copy=False
    )

    # 1-based → 0-based
    r = hsi[:, :, RED_BAND - 1]
    g = hsi[:, :, GREEN_BAND - 1]
    b = hsi[:, :, BLUE_BAND - 1]

    rgb = np.stack(
        [r, g, b],
        axis=-1
    )

    rgb_out = np.zeros_like(
        rgb,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 2-98 percentile stretch
    # --------------------------------------------------------

    for c in range(3):

        band = rgb[:, :, c]

        valid = np.isfinite(band)

        if not np.any(valid):
            continue

        low = np.percentile(
            band[valid],
            2
        )

        high = np.percentile(
            band[valid],
            98
        )

        if high <= low:

            rgb_out[:, :, c] = 0

        else:

            rgb_out[:, :, c] = (
                (band - low)
                / (high - low)
            )

    rgb_out = np.clip(
        rgb_out * 255,
        0,
        255
    ).astype(np.uint8)

    image = Image.fromarray(
        rgb_out,
        mode="RGB"
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image.save(
        str(output_png),
        format="PNG",
        dpi=(300, 300)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("STRATIFIED 100-PATCH SELECTION")
    print("=" * 100)

    print(f"Input : {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Seed  : {SEED}")
    print()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_tiff_root = (
        OUTPUT_ROOT / "selected_tiffs"
    )

    selected_png_root = (
        OUTPUT_ROOT / "fcc_png"
    )

    selected_tiff_root.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_png_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # STEP 1: FIND ALL TIFF FILES
    # ========================================================

    print("=" * 100)
    print("STEP 1/4: SCANNING TIFF FILES")
    print("=" * 100)

    all_records = []

    scan_start = time.time()

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_dir = INPUT_ROOT / split

        if not split_dir.exists():

            print(
                f"WARNING: missing {split_dir}"
            )

            continue

        tif_files = sorted(
            list(split_dir.rglob("*.tif"))
            + list(split_dir.rglob("*.tiff"))
        )

        print(
            f"{split.upper():5s}: "
            f"{len(tif_files):,} TIFF files"
        )

        for tif_path in tif_files:

            region, scene = get_scene_name(
                tif_path
            )

            all_records.append({
                "split": split,
                "region": region,
                "scene": scene,
                "path": str(tif_path),
                "filename": tif_path.name,
            })

    scan_elapsed = time.time() - scan_start

    print()
    print(
        f"Found {len(all_records):,} TIFF patches "
        f"in {format_time(scan_elapsed)}"
    )

    if len(all_records) == 0:

        raise RuntimeError(
            "No TIFF files found."
        )

    # ========================================================
    # STEP 2: CALCULATE STATISTICS
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 2/4: CALCULATING PATCH STATISTICS")
    print("=" * 100)
    print()
    print(
        "This is the slow step because every TIFF "
        "must be read."
    )
    print()

    records = []

    total = len(all_records)

    stats_start = time.time()

    for i, record in enumerate(
        all_records,
        start=1
    ):

        try:

            stats = calculate_patch_statistics(
                Path(record["path"])
            )

            record.update(stats)

            records.append(record)

        except Exception as e:

            print()
            print(
                f"\nERROR processing:"
            )
            print(record["path"])
            print(e)

        print_progress(
            current=i,
            total=total,
            start_time=stats_start,
            filename=record["filename"]
        )

    stats_elapsed = (
        time.time() - stats_start
    )

    print()
    print(
        f"Statistics completed in "
        f"{format_time(stats_elapsed)}"
    )

    print(
        f"Successful: {len(records):,}/{total:,}"
    )

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    all_stats_csv = (
        OUTPUT_ROOT /
        "all_patch_statistics.csv"
    )

    df.to_csv(
        all_stats_csv,
        index=False
    )

    print()
    print(
        f"All statistics saved to:\n"
        f"{all_stats_csv}"
    )

    # ========================================================
    # STEP 3: STRATIFIED SELECTION
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 3/4: SELECTING 70 / 10 / 20")
    print("=" * 100)

    rng = np.random.default_rng(
        SEED
    )

    selected_indices = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_df = df[
            df["split"] == split
        ].copy()

        target = TARGET_COUNTS[
            split
        ]

        if len(split_df) < target:

            raise RuntimeError(
                f"{split} contains only "
                f"{len(split_df)} patches."
            )

        print()
        print(
            f"{split.upper()}: "
            f"{len(split_df):,} available "
            f"→ {target} selected"
        )

        # ----------------------------------------------------
        # Scene-level proportional allocation
        # ----------------------------------------------------

        scene_groups = list(
            split_df.groupby(
                ["region", "scene"]
            )
        )

        scene_counts = [
            len(group)
            for _, group in scene_groups
        ]

        scene_allocations = (
            allocate_proportional(
                target,
                scene_counts
            )
        )

        split_selected = []

        for (
            (region_scene, scene_df),
            quota
        ) in zip(
            scene_groups,
            scene_allocations
        ):

            if quota <= 0:
                continue

            selected = select_from_scene(
                scene_df,
                int(quota),
                rng
            )

            split_selected.extend(
                selected
            )

        # ----------------------------------------------------
        # Safety correction
        # ----------------------------------------------------

        if len(split_selected) < target:

            remaining = list(
                set(split_df.index)
                - set(split_selected)
            )

            rng.shuffle(
                remaining
            )

            split_selected.extend(
                remaining[
                    :target - len(split_selected)
                ]
            )

        elif len(split_selected) > target:

            rng.shuffle(
                split_selected
            )

            split_selected = (
                split_selected[:target]
            )

        print(
            f"  Selected: "
            f"{len(split_selected)}"
        )

        selected_indices.extend(
            split_selected
        )

    # --------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------

    selected_df = df.loc[
        selected_indices
    ].copy()

    selected_df = selected_df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    selected_df.insert(
        0,
        "selection_id",
        range(
            1,
            len(selected_df) + 1
        )
    )

    # ========================================================
    # VERIFY
    # ========================================================

    print()
    print("=" * 100)
    print("VERIFYING FINAL SAMPLE")
    print("=" * 100)

    split_counts = (
        selected_df["split"]
        .value_counts()
        .reindex(
            ["train", "val", "test"]
        )
        .fillna(0)
        .astype(int)
    )

    print()
    print(split_counts)

    print()
    print(
        f"TOTAL = {len(selected_df)}"
    )

    if len(selected_df) != 100:
        raise RuntimeError(
            "Final selection is not exactly 100!"
        )

    if (
        split_counts["train"] != 70
        or split_counts["val"] != 10
        or split_counts["test"] != 20
    ):
        raise RuntimeError(
            "Train/Val/Test allocation is incorrect!"
        )

    # --------------------------------------------------------
    # Save final CSV
    # --------------------------------------------------------

    final_csv = (
        OUTPUT_ROOT /
        "selected_100_patches.csv"
    )

    selected_df.to_csv(
        final_csv,
        index=False
    )

    print()
    print(
        f"Selected patch list saved to:\n"
        f"{final_csv}"
    )

    # ========================================================
    # STEP 4: COPY TIFF + GENERATE PNG
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 4/4: COPYING TIFFs + GENERATING FCC PNGs")
    print("=" * 100)
    print()

    png_start = time.time()

    total_selected = len(
        selected_df
    )

    for i, (_, row) in enumerate(
        selected_df.iterrows(),
        start=1
    ):

        src = Path(
            row["path"]
        )

        split = row["split"]
        region = row["region"]
        scene = row["scene"]

        # ----------------------------------------------------
        # TIFF
        # ----------------------------------------------------

        tif_dir = (
            selected_tiff_root
            / split
            / region
            / scene
        )

        tif_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        dst_tif = (
            tif_dir /
            src.name
        )

        shutil.copy2(
            src,
            dst_tif
        )

        # ----------------------------------------------------
        # FCC PNG
        # ----------------------------------------------------

        png_dir = (
            selected_png_root
            / split
            / region
            / scene
        )

        dst_png = (
            png_dir /
            f"{src.stem}_FCC.png"
        )

        make_fcc_png(
            src,
            dst_png
        )

        print_progress(
            current=i,
            total=total_selected,
            start_time=png_start,
            filename=src.name
        )

    png_elapsed = (
        time.time() - png_start
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 100)
    print("COMPLETE")
    print("=" * 100)

    print()
    print("FINAL SAMPLE:")
    print(
        f"  Train      : {split_counts['train']}"
    )
    print(
        f"  Validation : {split_counts['val']}"
    )
    print(
        f"  Test       : {split_counts['test']}"
    )
    print(
        f"  TOTAL      : {len(selected_df)}"
    )

    print()
    print("FCC:")
    print(
        f"  RED   = Band {RED_BAND}"
    )
    print(
        f"  GREEN = Band {GREEN_BAND}"
    )
    print(
        f"  BLUE  = Band {BLUE_BAND}"
    )

    print()
    print("OUTPUT:")
    print(
        f"  {OUTPUT_ROOT}"
    )

    print()
    print("Important files:")
    print(
        f"  CSV : {final_csv}"
    )
    print(
        f"  Stats: {all_stats_csv}"
    )
    print(
        f"  TIFF: {selected_tiff_root}"
    )
    print(
        f"  PNG : {selected_png_root}"
    )

    print()
    print(
        f"PNG generation time: "
        f"{format_time(png_elapsed)}"
    )

    print()
    print("Random seed:", SEED)
    print("=" * 100)


if __name__ == "__main__":
    main()
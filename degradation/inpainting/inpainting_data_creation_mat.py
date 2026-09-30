#!/usr/bin/env python3

import os
import time
import gc
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np
import tifffile
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_ROOT = Path(
    "/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/inpaint_data/selected_tiffs"
)

OUTPUT_ROOT = Path(
    "/home/cvpr_pg_2/sudikin/data5_cvpr_pg_2/dataset/inpaint_data/total"
)

REAL_ROOT = OUTPUT_ROOT / "real"
IMAGE_ROOT = OUTPUT_ROOT / "image"

MASK2D_ROOT = OUTPUT_ROOT / "pngmask2d"
MASK3D_ROOT = OUTPUT_ROOT / "pngmask3d"

METADATA_CSV = OUTPUT_ROOT / "metadata.csv"
ERROR_LOG = OUTPUT_ROOT / "generation_errors.txt"


# ============================================================
# SETTINGS
# ============================================================

FORCED_BANDS = [12, 23, 43]

RED_BAND = 42
GREEN_BAND = 22
BLUE_BAND = 11

PNG_DPI = 300

CORRUPTION_TYPES = [
    "stripes",
    "deadline",
    "random",
    "blocks",
    "mixed",
    "mask50",
    "mask70",
    "mask90",
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for corr in CORRUPTION_TYPES:

    (REAL_ROOT / corr).mkdir(
        parents=True,
        exist_ok=True
    )

    (IMAGE_ROOT / corr).mkdir(
        parents=True,
        exist_ok=True
    )


(IMAGE_ROOT / "gt").mkdir(
    parents=True,
    exist_ok=True
)

for corr in CORRUPTION_TYPES:

    (MASK2D_ROOT / corr).mkdir(
        parents=True,
        exist_ok=True
    )

    (MASK3D_ROOT / corr).mkdir(
        parents=True,
        exist_ok=True
    )



# ============================================================
# LOAD ONE TIFF
# ============================================================

def load_one_tiff(file_path):

    """
    Load exactly ONE TIFF.

    Expected:
        Shape = (128, 128, 175)
        Axes  = YXS
        dtype = float32

    Therefore:
        Y = Height
        X = Width
        S = Spectral bands
    """

    with tifffile.TiffFile(file_path) as tif:

        if len(tif.series) == 0:
            raise ValueError(
                "TIFF contains no readable series."
            )

        series = tif.series[0]

        shape = series.shape
        axes = series.axes

        # ----------------------------------------------------
        # Verify TIFF organization
        # ----------------------------------------------------

        if axes != "YXS":

            raise ValueError(
                f"Unexpected TIFF axes: {axes}. "
                f"Expected YXS."
            )

        if len(shape) != 3:

            raise ValueError(
                f"Expected 3-D TIFF, got shape {shape}."
            )

        height = shape[0]
        width = shape[1]
        bands = shape[2]

        # ----------------------------------------------------
        # Verify spatial dimensions
        # ----------------------------------------------------

        if height != 128 or width != 128:

            raise ValueError(
                f"Unexpected spatial size: "
                f"{height}x{width}. "
                f"Expected 128x128."
            )

        # ----------------------------------------------------
        # Verify spectral dimensions
        # ----------------------------------------------------

        if bands != 175:

            raise ValueError(
                f"Unexpected number of bands: "
                f"{bands}. Expected 175."
            )

        # ----------------------------------------------------
        # Verify required FCC band
        # ----------------------------------------------------

        required_band = max(
            RED_BAND,
            GREEN_BAND,
            BLUE_BAND
        )

        if bands < required_band:

            raise ValueError(
                f"Only {bands} bands available, "
                f"but Band {required_band} is required."
            )

        # ----------------------------------------------------
        # Read ONLY this TIFF
        # ----------------------------------------------------

        image = series.asarray()

    return image.astype(
        np.float32
    )


# ============================================================
# FCC NORMALIZATION
# ============================================================

def normalize_band(band):

    band = np.asarray(
        band,
        dtype=np.float32
    )

    band = np.nan_to_num(
        band,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    p1 = np.percentile(
        band,
        1
    )

    p99 = np.percentile(
        band,
        99
    )

    if p99 > p1:

        band = (
            band - p1
        ) / (
            p99 - p1
        )

    else:

        bmin = np.min(band)
        bmax = np.max(band)

        if bmax > bmin:

            band = (
                band - bmin
            ) / (
                bmax - bmin
            )

        else:

            band = np.zeros_like(
                band
            )

    return np.clip(
        band,
        0.0,
        1.0
    )


# ============================================================
# CREATE FCC
# ============================================================

def create_fcc(image):

    """
    H x W x B -> RGB

    R = Band 42
    G = Band 22
    B = Band 11
    """

    if image.ndim != 3:

        raise ValueError(
            f"Expected 3-D HxWxB image, "
            f"got ndim={image.ndim}"
        )

    height, width, bands = image.shape

    if height != 128 or width != 128:

        raise ValueError(
            f"Expected 128x128 spatial image, "
            f"got {height}x{width}"
        )

    if bands != 175:

        raise ValueError(
            f"Expected 175 bands, "
            f"got {bands}"
        )

    red = image[
        :,
        :,
        RED_BAND - 1
    ]

    green = image[
        :,
        :,
        GREEN_BAND - 1
    ]

    blue = image[
        :,
        :,
        BLUE_BAND - 1
    ]

    red = normalize_band(red)
    green = normalize_band(green)
    blue = normalize_band(blue)

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    return rgb


# ============================================================
# FCC PNG
# ============================================================

def calculate_title_font_size(filename):

    filename_length = len(filename)

    if filename_length <= 30:
        return 28

    elif filename_length <= 45:
        return 24

    elif filename_length <= 60:
        return 20

    elif filename_length <= 80:
        return 16

    else:
        return 12


def save_fcc_png(
    image,
    output_path,
    filename
):

    rgb = create_fcc(
        image
    )

    title_font_size = (
        calculate_title_font_size(
            filename
        )
    )

    fig = plt.figure(
        figsize=(5, 5.6),
        facecolor="white"
    )

    ax = fig.add_axes(
        [
            0,
            0.08,
            1,
            0.82
        ]
    )

    ax.imshow(
        rgb,
        interpolation="nearest"
    )

    ax.axis("off")

    fig.text(
        0.5,
        0.94,
        filename,
        ha="center",
        va="center",
        fontsize=title_font_size
    )

    fig.savefig(
        output_path,
        dpi=PNG_DPI,
        facecolor="white"
    )

    plt.close(fig)

    del rgb


# ============================================================
# MASK PNG FUNCTIONS
# ============================================================

def save_mask2d_png(
    mask_3d,
    output_path,
    filename
):

    # EXACT LOGIC FROM PROVIDED CODE

    mask_2d = (
        mask_3d.mean(axis=2)
        > 0.999
    )

    plt.figure(
        figsize=(5, 5.6),
        dpi=PNG_DPI
    )

    plt.imshow(
        mask_2d,
        cmap="gray"
    )

    plt.title(
        filename
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=PNG_DPI,
        bbox_inches="tight"
    )

    plt.close()


def save_mask3d_png(
    mask_3d,
    output_path,
    filename
):

    # EXACT LOGIC FROM PROVIDED CODE

    mask_3d_plot = mask_3d.sum(
        axis=2
    )

    plt.figure(
        figsize=(5, 5.6),
        dpi=PNG_DPI
    )

    plt.imshow(
        mask_3d_plot,
        cmap="gray"
    )

    plt.title(
        filename
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=PNG_DPI,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# RANDOM LINE GENERATOR
# ============================================================

def draw_line(
    out,
    mask,
    band,
    angle,
    thickness,
    value=None,
    zero_all=False,
    used_signatures=None
):

    """
    Draw one line using the supplied angle.

    After drawing, each spectral band is checked independently.
    Only the 8-pixel image border is cleaned.

    Border cleanup is performed using 8-connected components:
        - component size <= 6  -> remove whole component
        - component size > 6   -> keep whole component

    Individual pixels are never removed from a valid line.
    """

    h, w, b = out.shape
    max_attempts = 1000

    for _attempt in range(max_attempts):

        if np.random.rand() < 0.5:

            x = np.random.randint(0, h)

            boundary = (
                "top"
                if np.random.rand() < 0.5
                else "bottom"
            )

            y = (
                0
                if boundary == "top"
                else w - 1
            )

        else:

            y = np.random.randint(0, w)

            boundary = (
                "left"
                if np.random.rand() < 0.5
                else "right"
            )

            x = (
                0
                if boundary == "left"
                else h - 1
            )

        xi0 = int(round(x))
        yi0 = int(round(y))

        signature = (
            boundary,
            xi0,
            yi0
        )

        if (
            used_signatures is None
            or signature not in used_signatures
        ):
            break

    else:
        signature = None

    if used_signatures is not None and signature is not None:
        used_signatures.add(signature)

    dx = np.cos(angle)
    dy = np.sin(angle)

    scale = max(
        abs(dx),
        abs(dy)
    )

    dx /= scale
    dy /= scale

    if value is None:
        value = np.random.uniform(
            100,
            3000
        )

    # ------------------------------------------------------------
    # Save original values of pixels touched by this line
    # ------------------------------------------------------------

    original_values = {}

    # Keep the exact spatial pixels generated by THIS line.
    line_pixels = []

    for _ in range(max(h, w) * 2):

        xi = int(round(x))
        yi = int(round(y))

        if (
            0 <= xi < h
            and
            0 <= yi < w
        ):

            if (xi, yi) not in original_values:

                if zero_all:

                    original_values[
                        (xi, yi)
                    ] = out[xi, yi, :].copy()

                else:

                    original_values[
                        (xi, yi)
                    ] = out[xi, yi, band].copy()

            if not line_pixels or line_pixels[-1] != (xi, yi):
                line_pixels.append((xi, yi))

            if zero_all:

                out[
                    xi,
                    yi,
                    :
                ] = 0

                mask[
                    xi,
                    yi,
                    :
                ] = 1

            else:

                out[
                    xi,
                    yi,
                    band
                ] = value

                mask[
                    xi,
                    yi,
                    band
                ] = 1

        x += dx
        y += dy

    # ============================================================
    # BORDER CLEANUP
    # ============================================================

    border = np.zeros(
        (h, w),
        dtype=bool
    )

    border[:8, :] = True
    border[-8:, :] = True
    border[:, :8] = True
    border[:, -8:] = True

    # ------------------------------------------------------------
    # Check each spectral band independently
    # ------------------------------------------------------------

    bands_to_check = (
        range(b)
        if zero_all
        else [band]
    )

    for band_idx in bands_to_check:

        # Current line mask for this spectral band only
        line_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        for xi, yi in line_pixels:
            line_mask[xi, yi] = 1

        visited = np.zeros(
            (h, w),
            dtype=bool
        )

        # --------------------------------------------------------
        # Find 8-connected components
        # --------------------------------------------------------

        for xi, yi in line_pixels:

            if visited[xi, yi]:
                continue

            if line_mask[xi, yi] == 0:
                continue

            component = []
            stack = [(xi, yi)]
            visited[xi, yi] = True

            while stack:

                cx, cy = stack.pop()

                component.append(
                    (cx, cy)
                )

                for dx8 in (-1, 0, 1):

                    for dy8 in (-1, 0, 1):

                        if (
                            dx8 == 0
                            and
                            dy8 == 0
                        ):
                            continue

                        nx = cx + dx8
                        ny = cy + dy8

                        if not (
                            0 <= nx < h
                            and
                            0 <= ny < w
                        ):
                            continue

                        if visited[nx, ny]:
                            continue

                        if line_mask[nx, ny] == 0:
                            continue

                        visited[nx, ny] = True

                        stack.append(
                            (nx, ny)
                        )

            # ----------------------------------------------------
            # Only clean components touching the 8-pixel border
            # ----------------------------------------------------

            touches_border = any(
                border[cx, cy]
                for cx, cy in component
            )

            if not touches_border:
                continue

            # ----------------------------------------------------
            # Small border component -> remove WHOLE component
            # ----------------------------------------------------

            if len(component) <= 6:

                for cx, cy in component:

                    mask[
                        cx,
                        cy,
                        band_idx
                    ] = 0

                    if (cx, cy) in original_values:

                        if zero_all:

                            out[
                                cx,
                                cy,
                                :
                            ] = original_values[
                                (cx, cy)
                            ]

                        elif band_idx == band:

                            out[
                                cx,
                                cy,
                                band
                            ] = original_values[
                                (cx, cy)
                            ]

    return {
        "angle_degrees": float(
            np.rad2deg(angle)
        ),
        "boundary": boundary,
        "start_x": xi0,
        "start_y": yi0,
        "thickness": int(thickness)
    }
# ============================================================
# STRIPES
# ============================================================

def mask_stripes(
    data,
    used_signatures
):

    out = data.copy()

    mask = np.zeros_like(
        data
    )

    h, w, b = data.shape

    # ONE angle for all stripe lines in this patch.
    stripe_angle_degrees = np.random.uniform(
        160.0,
        170.0
    )

    stripe_angle = np.deg2rad(
        stripe_angle_degrees
    )

    thickness = np.random.choice(
        [1, 2]
    )

    line_records = []

    for band in range(b):

        apply = (
            band in FORCED_BANDS
        ) or (
            np.random.rand() < 0.4
        )

        if apply:

            n_lines = np.random.randint(
                1,
                5
            )

            for _ in range(n_lines):

                record = draw_line(
                    out=out,
                    mask=mask,
                    band=band,
                    angle=stripe_angle,
                    thickness=thickness,
                    used_signatures=used_signatures
                )

                line_records.append(
                    record
                )

    # Guarantee at least one stripe.
    if len(line_records) == 0:

        band = np.random.choice(
            FORCED_BANDS
        )

        record = draw_line(
            out=out,
            mask=mask,
            band=band,
            angle=stripe_angle,
            thickness=thickness,
            used_signatures=used_signatures
        )

        line_records.append(
            record
        )

    return out, mask, line_records


# ============================================================
# DEADLINE
# ============================================================

def mask_deadline(
    data,
    used_signatures
):

    out = data.copy()

    mask = np.zeros_like(
        data
    )

    # ONE angle for all deadline lines in this patch.
    deadline_angle_degrees = np.random.uniform(
        160.0,
        170.0
    )

    deadline_angle = np.deg2rad(
        deadline_angle_degrees
    )

    thickness = np.random.choice(
        [1, 2]
    )

    line_records = []

    n_lines = np.random.randint(
        2,
        5
    )

    for _ in range(n_lines):

        record = draw_line(
            out=out,
            mask=mask,
            band=None,
            angle=deadline_angle,
            thickness=thickness,
            zero_all=True,
            used_signatures=used_signatures
        )

        line_records.append(
            record
        )

    return out, mask, line_records


# ============================================================
# RANDOM
# ============================================================

def mask_random(data):

    out = data.copy()

    mask = np.zeros_like(
        data
    )

    rand_mask = (
        np.random.rand(
            *data.shape
        )
        < 0.2
    )

    for band in FORCED_BANDS:

        rand_mask[
            :,
            :,
            band
        ] = (
            np.random.rand(
                data.shape[0],
                data.shape[1]
            )
            < 0.3
        )

    noise = np.random.uniform(
        0.1,
        0.7,
        size=data.shape
    )

    out[
        rand_mask
    ] *= noise[
        rand_mask
    ]

    mask[
        rand_mask
    ] = 1

    return out, mask


# ============================================================
# BLOCKS
# ============================================================

def mask_blocks(data):

    out = data.copy()

    mask = np.zeros_like(
        data
    )

    # EXACT HWB
    h, w, b = data.shape

    for band in range(b):

        apply = (
            band in FORCED_BANDS
        ) or (
            np.random.rand()
            < 0.5
        )

        if apply:

            bh = np.random.randint(
                h // 8,
                h // 3
            )

            bw = np.random.randint(
                w // 8,
                w // 3
            )

            x = np.random.randint(
                0,
                h - bh
            )

            y = np.random.randint(
                0,
                w - bw
            )

            factor = np.random.choice(
                [
                    np.random.uniform(
                        0.1,
                        0.4
                    ),

                    np.random.uniform(
                        0.4,
                        0.8
                    ),

                    0
                ]
            )

            out[
                x:x + bh,
                y:y + bw,
                band
            ] *= factor

            mask[
                x:x + bh,
                y:y + bw,
                band
            ] = 1

    return out, mask


# ============================================================
# MIXED
# ============================================================

def mixed_from_existing(
    gt,
    ms,
    md,
    mr,
    mb
):

    # EXACT UNION LOGIC

    m_union = np.clip(
        ms + md + mr + mb,
        0,
        1
    )

    out = gt.copy()

    # --------------------------------------------------------
    # Dead
    # --------------------------------------------------------

    out[
        md == 1
    ] = 0

    # --------------------------------------------------------
    # Stripes
    # --------------------------------------------------------

    rand_vals = np.random.uniform(
        100,
        3000,
        size=gt.shape
    )

    cond = (
        (ms == 1)
        &
        (md == 0)
    )

    out[
        cond
    ] = rand_vals[
        cond
    ]

    # --------------------------------------------------------
    # Random
    # --------------------------------------------------------

    noise = np.random.uniform(
        0.1,
        0.7,
        size=gt.shape
    )

    cond = (
        (mr == 1)
        &
        (md == 0)
    )

    out[
        cond
    ] *= noise[
        cond
    ]

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    factors = np.random.uniform(
        0,
        0.8,
        size=gt.shape
    )

    cond = (
        (mb == 1)
        &
        (md == 0)
    )

    out[
        cond
    ] *= factors[
        cond
    ]

    return out, m_union


# ============================================================
# PERCENT MASK
# ============================================================

def generate_percent_mask(
    gt,
    percent
):

    # EXACT HWB

    h, w, b = gt.shape

    mask2d = (
        np.random.rand(
            h,
            w
        )
        < percent
    )

    mask3d = np.repeat(
        mask2d[:, :, None],
        b,
        axis=2
    )

    out = gt.copy()

    out[
        mask3d == 1
    ] = 0

    return out, mask3d


# ============================================================
# SAVE MAT
# ============================================================

def save_mat(
    mat_path,
    gt,
    mask_3d,
    corrupted
):

    mask_2d = (
        mask_3d.mean(axis=2)
        > 0.999
    ).astype(
        np.float32
    )

    sio.savemat(
        str(mat_path),
        {
            "image":
                gt.astype(np.float32),

            "mask_3d":
                mask_3d.astype(np.float32),

            "mask_2d":
                mask_2d,

            "corrupted":
                corrupted.astype(np.float32)
        },
        do_compression=False
    )


# ============================================================
# PROCESS ONE TIFF
# ============================================================

def process_one_tiff(args):

    # IMPORTANT:
    # enumerate(tiffs) produces:
    # (image_index, file_path)

    image_index, file_path = args

    file_path = Path(file_path)

    try:

        # ====================================================
        # LOAD
        # ====================================================

        gt = load_one_tiff(
            str(file_path)
        )

        # ====================================================
        # EXACT H W B
        # ====================================================

        h, w, b = gt.shape

        if (
            h != 128
            or w != 128
            or b != 175
        ):
            raise ValueError(
                f"Unexpected shape "
                f"{gt.shape}. "
                f"Expected (128, 128, 175)."
            )

        # ====================================================
        # UNIQUE RANDOM STREAM FOR THIS IMAGE
        # ====================================================

        seed_sequence = np.random.SeedSequence(
            [
                20260920,
                int(image_index),
                int(time.time_ns() % (2**32))
            ]
        )

        rng_seed = seed_sequence.generate_state(
            1,
            dtype=np.uint32
        )[0]

        np.random.seed(
            int(rng_seed)
        )

        # ====================================================
        # USED LINE SIGNATURES
        #
        # Prevent identical line configuration within
        # this image.
        # ====================================================

        used_signatures = set()

        # ====================================================
        # STRIPES
        # ====================================================

        stripes, mask_stripe, stripe_records = (
            mask_stripes(
                gt,
                used_signatures
            )
        )

        # ====================================================
        # DEADLINE
        # ====================================================

        deadline, mask_dead, dead_records = (
            mask_deadline(
                gt,
                used_signatures
            )
        )

        # ====================================================
        # RANDOM
        # ====================================================

        random_corrupted, mask_random_3d = (
            mask_random(
                gt
            )
        )

        # ====================================================
        # BLOCKS
        # ====================================================

        blocks, mask_blocks_3d = (
            mask_blocks(
                gt
            )
        )

        # ====================================================
        # MIXED
        # USE EXACT SAME MASKS
        # ====================================================

        mixed, mask_mixed = (
            mixed_from_existing(
                gt,
                mask_stripe,
                mask_dead,
                mask_random_3d,
                mask_blocks_3d
            )
        )

        # ====================================================
        # PERCENTAGE MASKS
        # ====================================================

        mask50_corrupted, mask50 = (
            generate_percent_mask(
                gt,
                0.5
            )
        )

        mask70_corrupted, mask70 = (
            generate_percent_mask(
                gt,
                0.7
            )
        )

        mask90_corrupted, mask90 = (
            generate_percent_mask(
                gt,
                0.9
            )
        )

        # ====================================================
        # CASES
        # ====================================================

        cases = {

            "stripes": (
                mask_stripe,
                stripes
            ),

            "deadline": (
                mask_dead,
                deadline
            ),

            "random": (
                mask_random_3d,
                random_corrupted
            ),

            "blocks": (
                mask_blocks_3d,
                blocks
            ),

            "mixed": (
                mask_mixed,
                mixed
            ),

            "mask50": (
                mask50,
                mask50_corrupted
            ),

            "mask70": (
                mask70,
                mask70_corrupted
            ),

            "mask90": (
                mask90,
                mask90_corrupted
            ),
        }

        # ====================================================
        # SOURCE FILENAME
        # ====================================================

        filename_stem = file_path.stem

        # ====================================================
        # GT FCC
        # ====================================================

        gt_png_path = (
            IMAGE_ROOT
            / "gt"
            / f"{filename_stem}.png"
        )

        save_fcc_png(
            gt,
            gt_png_path,
            f"{filename_stem}.png"
        )

        # ====================================================
        # METADATA
        # ====================================================

        rows = []

        # ====================================================
        # SAVE EACH CASE
        # ====================================================

        for corr, (
            mask_3d,
            corrupted
        ) in cases.items():

            # ------------------------------------------------
            # <filename>_<corr>
            # ------------------------------------------------

            case_stem = (
                f"{filename_stem}_{corr}"
            )

            # ------------------------------------------------
            # MAT
            # ------------------------------------------------

            mat_path = (
                REAL_ROOT
                / corr
                / f"{case_stem}.mat"
            )

            # ------------------------------------------------
            # FCC PNG
            # ------------------------------------------------

            png_path = (
                IMAGE_ROOT
                / corr
                / f"{case_stem}.png"
            )

            # ------------------------------------------------
            # MASK PNGs
            # ------------------------------------------------

            mask2d_png_path = (
                MASK2D_ROOT
                / corr
                / f"{case_stem}.png"
            )

            mask3d_png_path = (
                MASK3D_ROOT
                / corr
                / f"{case_stem}.png"
            )

            # ------------------------------------------------
            # SAVE MAT
            # ------------------------------------------------

            save_mat(
                mat_path,
                gt,
                mask_3d,
                corrupted
            )

            # ------------------------------------------------
            # SAVE FCC
            # ------------------------------------------------

            save_fcc_png(
                corrupted,
                png_path,
                f"{case_stem}.png"
            )

            # ------------------------------------------------
            # SAVE MASK 2D
            # ------------------------------------------------

            save_mask2d_png(
                mask_3d,
                mask2d_png_path,
                f"{case_stem}.png"
            )

            # ------------------------------------------------
            # SAVE MASK 3D
            # ------------------------------------------------

            save_mask3d_png(
                mask_3d,
                mask3d_png_path,
                f"{case_stem}.png"
            )

            # ------------------------------------------------
            # MASK STATISTICS
            # ------------------------------------------------

            mask_voxels = int(
                np.sum(
                    mask_3d > 0
                )
            )

            total_voxels = int(
                mask_3d.size
            )

            mask_percentage = (
                mask_voxels
                / total_voxels
                * 100.0
            )

            # EXACT MASK2D DEFINITION
            mask_2d = (
                mask_3d.mean(axis=2)
                > 0.999
            ).astype(
                np.float32
            )

            mask_2d_pixels = int(
                np.sum(
                    mask_2d > 0
                )
            )

            total_spatial_pixels = (
                h * w
            )

            mask_2d_percentage = (
                mask_2d_pixels
                / total_spatial_pixels
                * 100.0
            )

            # ------------------------------------------------
            # METADATA ROW
            # ------------------------------------------------

            rows.append(
                {
                    "filename":
                        f"{case_stem}.mat",

                    "filename_stem":
                        case_stem,

                    "input_full_path":
                        str(file_path),

                    "corr":
                        corr,

                    "mat_path":
                        str(mat_path),

                    "png_path":
                        str(png_path),

                    "height":
                        h,

                    "width":
                        w,

                    "bands":
                        b,

                    "dtype":
                        str(gt.dtype),

                    "mask_voxels":
                        mask_voxels,

                    "total_voxels":
                        total_voxels,

                    "mask_percentage":
                        mask_percentage,

                    "mask_2d_pixels":
                        mask_2d_pixels,

                    "total_spatial_pixels":
                        total_spatial_pixels,

                    "mask_2d_percentage":
                        mask_2d_percentage,
                }
            )

        # ====================================================
        # CLEANUP
        # ====================================================

        del gt

        del stripes
        del mask_stripe

        del deadline
        del mask_dead

        del random_corrupted
        del mask_random_3d

        del blocks
        del mask_blocks_3d

        del mixed
        del mask_mixed

        del mask50_corrupted
        del mask50

        del mask70_corrupted
        del mask70

        del mask90_corrupted
        del mask90

        gc.collect()

        return (
            True,
            str(file_path),
            rows,
            None
        )

    except Exception as exc:

        return (
            False,
            str(file_path),
            [],
            repr(exc)
        )


# ============================================================
# DISCOVER TIFFS
# ============================================================

def discover_tiffs():

    tiffs = []

    for path in INPUT_ROOT.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in {
                ".tif",
                ".tiff"
            }
        ):

            tiffs.append(
                path
            )

    return sorted(
        tiffs
    )


# ============================================================
# PROGRESS
# ============================================================

def print_progress(
    processed,
    total,
    successful,
    failed,
    start_time
):

    elapsed = (
        time.time()
        -
        start_time
    )

    speed = (
        processed / elapsed
        if elapsed > 0
        else 0
    )

    remaining = (
        total - processed
    )

    eta = (
        remaining / speed
        if speed > 0
        else 0
    )

    percentage = (
        processed
        /
        total
        *
        100
    )

    bar_width = 40

    filled = int(
        bar_width
        *
        processed
        /
        total
    )

    if filled < bar_width:

        bar = (
            "=" * filled
            +
            ">"
            +
            " " *
            (
                bar_width
                -
                filled
                -
                1
            )
        )

    else:

        bar = "=" * bar_width

    elapsed_h = int(
        elapsed // 3600
    )

    elapsed_m = int(
        (elapsed % 3600) // 60
    )

    elapsed_s = int(
        elapsed % 60
    )

    eta_h = int(
        eta // 3600
    )

    eta_m = int(
        (eta % 3600) // 60
    )

    eta_s = int(
        eta % 60
    )

    print(
        f"\r[{bar}] "
        f"{percentage:6.2f}% | "
        f"{processed}/{total} | "
        f"OK:{successful} | "
        f"FAIL:{failed} | "
        f"{speed:.2f} TIFF/s | "
        f"Elapsed: "
        f"{elapsed_h:02d}:"
        f"{elapsed_m:02d}:"
        f"{elapsed_s:02d} | "
        f"ETA: "
        f"{eta_h:02d}:"
        f"{eta_m:02d}:"
        f"{eta_s:02d}",
        end="",
        flush=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print(
        "=" * 75
    )

    print(
        "HYPERSPECTRAL INPAINTING DATASET GENERATION"
    )

    print(
        "=" * 75
    )

    print(
        f"Input       : {INPUT_ROOT}"
    )

    print(
        f"Output      : {OUTPUT_ROOT}"
    )

    print()

    # ========================================================
    # DISCOVER
    # ========================================================

    tiffs = discover_tiffs()

    total = len(
        tiffs
    )

    print(
        f"Found {total} TIFF files."
    )

    print()

    if total == 0:

        print(
            "No TIFF files found."
        )

        return

    # ========================================================
    # INDEX EACH IMAGE
    # ========================================================

    indexed_tiffs = list(
        enumerate(tiffs)
    )

    # ========================================================
    # CPU WORKERS
    # ========================================================

    workers = max(
        1,
        mp.cpu_count() - 1
    )

    print(
        f"CPU workers: {workers}"
    )

    print()

    # ========================================================
    # CSV FIELDS
    # ========================================================

    fieldnames = [

        "filename",

        "filename_stem",

        "input_full_path",

        "corr",

        "mat_path",

        "png_path",

        "height",

        "width",

        "bands",

        "dtype",

        "mask_voxels",

        "total_voxels",

        "mask_percentage",

        "mask_2d_pixels",

        "total_spatial_pixels",

        "mask_2d_percentage",
    ]

    all_metadata = []

    errors = []

    # ========================================================
    # PROCESS
    # ========================================================

    with mp.Pool(
        processes=workers
    ) as pool:

        for processed, result in enumerate(
            pool.imap_unordered(
                process_one_tiff,
                indexed_tiffs
            ),
            start=1
        ):

            success, path, rows, error = (
                result
            )

            if success:

                all_metadata.extend(
                    rows
                )

            else:

                errors.append(
                    (
                        path,
                        error
                    )
                )

            successful = (
                processed
                -
                len(errors)
            )

            failed = len(
                errors
            )

            print_progress(
                processed,
                total,
                successful,
                failed,
                start_time
            )

    print()
    print()

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        METADATA_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_metadata
        )

    # ========================================================
    # SAVE ERRORS
    # ========================================================

    if errors:

        with open(
            ERROR_LOG,
            "w"
        ) as f:

            for path, error in errors:

                f.write(
                    f"{path}\n"
                )

                f.write(
                    f"{error}\n"
                )

                f.write(
                    "-" * 80
                    +
                    "\n"
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    elapsed = (
        time.time()
        -
        start_time
    )

    print(
        "=" * 75
    )

    print(
        "GENERATION COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"TIFF files    : {total}"
    )

    print(
        f"Successful    : "
        f"{total - len(errors)}"
    )

    print(
        f"Failed        : "
        f"{len(errors)}"
    )

    print(
        f"Metadata rows : "
        f"{len(all_metadata)}"
    )

    print(
        f"Metadata CSV  : "
        f"{METADATA_CSV}"
    )

    if errors:

        print(
            f"Error log     : "
            f"{ERROR_LOG}"
        )

    print(
        f"Elapsed       : "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
"""Generate an inline, axis-free OSU pseudospectrum spelling Jonathan Lindbloom.

The construction follows the original name-art generator in this directory:
sample eigenvalues inside bold text, couple nearby eigenvalues in local 2x2
upper-triangular blocks, and plot contours of sigma_min(z I - A).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple


# Keep Matplotlib's cache out of the repository and use a Windows-safe path.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "jonathan_inline_pseudospectrum_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SITE_DIR = SCRIPT_DIR.parents[2]

OUTPUT_STEM = "jonathan_lindbloom_inline_osu_pseudospectrum"
OUTPUT_PNG = SCRIPT_DIR / f"{OUTPUT_STEM}.png"
OUTPUT_MATRIX = SCRIPT_DIR / f"{OUTPUT_STEM}_matrix.npy"
SITE_OUTPUT_PNG = SITE_DIR / "assets" / "images" / "jonathan-lindbloom-pseudospectrum-inline-osu.png"
MEDIUM_OUTPUT_STEM = f"{OUTPUT_STEM}_medium_inner_contours"
MEDIUM_OUTPUT_PNG = SCRIPT_DIR / f"{MEDIUM_OUTPUT_STEM}.png"
MEDIUM_SITE_OUTPUT_PNG = (
    SITE_DIR
    / "assets"
    / "images"
    / "jonathan-lindbloom-pseudospectrum-inline-osu-medium-inner-contours.png"
)
DENSE_OUTPUT_STEM = f"{OUTPUT_STEM}_dense_inner_contours"
DENSE_OUTPUT_PNG = SCRIPT_DIR / f"{DENSE_OUTPUT_STEM}.png"
DENSE_SITE_OUTPUT_PNG = (
    SITE_DIR
    / "assets"
    / "images"
    / "jonathan-lindbloom-pseudospectrum-inline-osu-dense-inner-contours.png"
)

FIRST_NAME = "Jonathan"
LAST_NAME = "Lindbloom"
FONT = FontProperties(family="DejaVu Sans", weight="bold")
TARGET_WIDTH = 14.0
WORD_GAP_EM = 0.62
SAMPLE_PITCH = 0.12
NONNORMAL_COUPLING = 0.34
BASE_EPSILON = 0.092

DEFAULT_GRID_SIZE_X = 1400
DEFAULT_GRID_SIZE_Y = 420
FIGURE_WIDTH_INCHES = 16.0

# Inner to outer contour levels. The low values sit closest to the eigenvalues.
LEVEL_FACTORS = np.array([0.35, 0.50, 0.72, 1.00, 1.45, 2.15, 3.30, 5.20, 8.20])

# Middle-density alternative: 12 contours, with the three additions kept near
# the small-epsilon center of the plot.
MEDIUM_INNER_LEVEL_FACTORS = np.array(
    [0.18, 0.26, 0.35, 0.50, 0.72, 1.00, 1.25, 1.45, 2.15, 3.30, 5.20, 8.20]
)

# Alternative treatment: most additional lines are concentrated below 1.5 eps_0.
DENSE_INNER_LEVEL_FACTORS = np.array(
    [0.12, 0.16, 0.21, 0.27, 0.35, 0.45, 0.58, 0.72, 0.88, 1.05, 1.30, 1.65, 2.15, 3.30, 5.20, 8.20]
)

# Official Ohio State primary/shade values plus permitted black, arranged with
# strictly increasing luminance so the center is unambiguously darkest.
# https://bux.osu.edu/color/primary-colors/
OSU_CONTOUR_COLORS = (
    "#000000",  # black
    "#4A0513",  # scarlet dark 60
    "#70071C",  # scarlet dark 40
    "#BA0C2F",  # scarlet
    "#3F4443",  # gray dark 60
    "#646A6E",  # gray dark 40
    "#868E92",  # gray dark 20
    "#A7B1B7",  # gray
    "#DFE3E5",  # gray light 60
)

MEDIUM_INNER_CONTOUR_COLORS = (
    "#000000",
    "#30030C",
    "#4A0513",
    "#70071C",
    "#8D0924",
    "#BA0C2F",
    "#3F4443",
    "#515659",
    "#646A6E",
    "#868E92",
    "#A7B1B7",
    "#DFE3E5",
)

# A longer, monotone dark-scarlet ramp gives each dense inner contour its own
# shade while retaining the same official scarlet/gray anchors as the base art.
DENSE_INNER_CONTOUR_COLORS = (
    "#000000",
    "#100104",
    "#200208",
    "#30030C",
    "#400410",
    "#4A0513",
    "#5B0618",
    "#70071C",
    "#8D0924",
    "#A90B2B",
    "#BA0C2F",
    "#3F4443",
    "#646A6E",
    "#868E92",
    "#A7B1B7",
    "#DFE3E5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300).")
    parser.add_argument(
        "--grid-x",
        type=int,
        default=DEFAULT_GRID_SIZE_X,
        help=f"Horizontal grid samples (default: {DEFAULT_GRID_SIZE_X}).",
    )
    parser.add_argument(
        "--grid-y",
        type=int,
        default=DEFAULT_GRID_SIZE_Y,
        help=f"Vertical grid samples (default: {DEFAULT_GRID_SIZE_Y}).",
    )
    parser.add_argument(
        "--skip-site-copy",
        action="store_true",
        help="Do not copy the final PNG into new_site/assets/images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi <= 0 or args.grid_x < 200 or args.grid_y < 100:
        raise ValueError("dpi must be positive; grid-x/grid-y must be at least 200/100")

    name_path = build_inline_name_path()
    eigenvalues = sample_points_in_path(name_path)
    pairs, singles = build_local_pairs(eigenvalues)
    matrix = build_nonnormal_matrix(eigenvalues, pairs, singles, NONNORMAL_COUPLING)
    np.save(OUTPUT_MATRIX, matrix)

    levels = BASE_EPSILON * LEVEL_FACTORS
    medium_inner_levels = BASE_EPSILON * MEDIUM_INNER_LEVEL_FACTORS
    dense_inner_levels = BASE_EPSILON * DENSE_INNER_LEVEL_FACTORS
    xlim, ylim = plotting_limits(name_path, dense_inner_levels)
    x_grid, y_grid, z_grid = complex_grid(xlim, ylim, args.grid_x, args.grid_y)
    sigma_min = block_nonnormal_singular_value_grid(
        eigenvalues,
        pairs,
        singles,
        NONNORMAL_COUPLING,
        z_grid,
    )

    save_osu_contour_plot(
        x_grid,
        y_grid,
        sigma_min,
        xlim,
        ylim,
        levels,
        OSU_CONTOUR_COLORS,
        (0.88, 1.58),
        OUTPUT_PNG,
        args.dpi,
    )
    save_osu_contour_plot(
        x_grid,
        y_grid,
        sigma_min,
        xlim,
        ylim,
        medium_inner_levels,
        MEDIUM_INNER_CONTOUR_COLORS,
        (0.52, 1.50),
        MEDIUM_OUTPUT_PNG,
        args.dpi,
    )
    save_osu_contour_plot(
        x_grid,
        y_grid,
        sigma_min,
        xlim,
        ylim,
        dense_inner_levels,
        DENSE_INNER_CONTOUR_COLORS,
        (0.62, 1.42),
        DENSE_OUTPUT_PNG,
        args.dpi,
    )

    if not args.skip_site_copy:
        SITE_OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PNG, SITE_OUTPUT_PNG)
        shutil.copy2(MEDIUM_OUTPUT_PNG, MEDIUM_SITE_OUTPUT_PNG)
        shutil.copy2(DENSE_OUTPUT_PNG, DENSE_SITE_OUTPUT_PNG)

    print(f"eigenvalues: {len(eigenvalues)} ({len(pairs)} pairs, {len(singles)} singles)")
    print(f"wrote {OUTPUT_MATRIX}")
    print(f"wrote {OUTPUT_PNG}")
    print(f"wrote {MEDIUM_OUTPUT_PNG}")
    print(f"wrote {DENSE_OUTPUT_PNG}")
    if not args.skip_site_copy:
        print(f"wrote {SITE_OUTPUT_PNG}")
        print(f"wrote {MEDIUM_SITE_OUTPUT_PNG}")
        print(f"wrote {DENSE_SITE_OUTPUT_PNG}")


def build_inline_name_path() -> MplPath:
    """Return one baseline-aligned path with a deliberate inter-word gap."""

    first = TextPath((0.0, 0.0), FIRST_NAME, size=1.0, prop=FONT)
    last = TextPath((0.0, 0.0), LAST_NAME, size=1.0, prop=FONT)
    first_ext = first.get_extents()
    last_ext = last.get_extents()

    last = last.transformed(
        Affine2D().translate(first_ext.x1 - last_ext.x0 + WORD_GAP_EM, 0.0)
    )
    vertices = np.concatenate([first.vertices, last.vertices], axis=0)
    codes = np.concatenate([first.codes, last.codes], axis=0)
    combined = MplPath(vertices, codes)

    ext = combined.get_extents()
    scale = TARGET_WIDTH / (ext.x1 - ext.x0)
    combined = combined.transformed(Affine2D().scale(scale, scale))
    ext = combined.get_extents()
    return combined.transformed(
        Affine2D().translate(-0.5 * (ext.x0 + ext.x1), -0.5 * (ext.y0 + ext.y1))
    )


def sample_points_in_path(path: MplPath) -> np.ndarray:
    """Sample a regular eigenvalue lattice inside the glyphs, preserving counters."""

    ext = path.get_extents()
    xs = np.arange(ext.x0, ext.x1 + SAMPLE_PITCH, SAMPLE_PITCH)
    ys = np.arange(ext.y0, ext.y1 + SAMPLE_PITCH, SAMPLE_PITCH)
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    inside = np.zeros(len(points), dtype=bool)
    for polygon in path.to_polygons(closed_only=True):
        if len(polygon) < 3:
            continue
        inside ^= MplPath(polygon, closed=True).contains_points(points, radius=0.0)

    selected = points[inside]
    if not len(selected):
        raise RuntimeError("The text path produced no sampled eigenvalues.")
    return selected[:, 0] + 1j * selected[:, 1]


def build_local_pairs(eigenvalues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pair nearby row-wise eigenvalues into local nonnormal 2x2 blocks."""

    points = np.column_stack([eigenvalues.real, eigenvalues.imag])
    used = np.zeros(len(eigenvalues), dtype=bool)
    pair_list = []
    leftovers = []

    for y_value in np.unique(points[:, 1]):
        row_indices = np.flatnonzero(points[:, 1] == y_value)
        row_indices = row_indices[np.argsort(points[row_indices, 0])]
        stop = len(row_indices) - (len(row_indices) % 2)
        for index in range(0, stop, 2):
            left = int(row_indices[index])
            right = int(row_indices[index + 1])
            pair_list.append((left, right))
            used[left] = True
            used[right] = True
        if len(row_indices) % 2:
            leftovers.append(int(row_indices[-1]))

    if leftovers:
        leftover_indices = np.asarray(leftovers, dtype=int)
        leftover_indices = leftover_indices[
            np.argsort(points[leftover_indices, 0], kind="mergesort")
        ]
        still_unpaired = []
        for x_value in np.unique(points[leftover_indices, 0]):
            column_indices = leftover_indices[points[leftover_indices, 0] == x_value]
            column_indices = column_indices[np.argsort(points[column_indices, 1])]
            stop = len(column_indices) - (len(column_indices) % 2)
            for index in range(0, stop, 2):
                lower = int(column_indices[index])
                upper = int(column_indices[index + 1])
                pair_list.append((lower, upper))
                used[lower] = True
                used[upper] = True
            if len(column_indices) % 2:
                still_unpaired.append(int(column_indices[-1]))
        leftovers = still_unpaired

    singles = np.flatnonzero(~used)
    if leftovers:
        leftover_array = np.asarray(leftovers, dtype=int)
        leftover_array = leftover_array[~used[leftover_array]]
        if len(leftover_array):
            singles = np.unique(np.concatenate([singles, leftover_array]))

    pairs = np.asarray(pair_list, dtype=int).reshape((-1, 2))
    return pairs, singles.astype(int)


def build_nonnormal_matrix(
    eigenvalues: np.ndarray,
    pairs: np.ndarray,
    singles: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """Assemble the block-diagonal nonnormal matrix used by the contour plot."""

    matrix = np.zeros((len(eigenvalues), len(eigenvalues)), dtype=np.complex128)
    cursor = 0
    for left, right in pairs:
        matrix[cursor, cursor] = eigenvalues[left]
        matrix[cursor, cursor + 1] = coupling
        matrix[cursor + 1, cursor + 1] = eigenvalues[right]
        cursor += 2
    for index in singles:
        matrix[cursor, cursor] = eigenvalues[index]
        cursor += 1
    return matrix


def plotting_limits(
    path: MplPath,
    contour_levels: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Pad the text for the outer halo without recreating the old tall canvas."""

    ext = path.get_extents()
    halo = float(np.max(contour_levels)) / BASE_EPSILON * SAMPLE_PITCH * 0.62
    x_pad = 0.72 + 1.15 * halo
    y_pad = 0.48 + 1.15 * halo
    return (ext.x0 - x_pad, ext.x1 + x_pad), (ext.y0 - y_pad, ext.y1 + y_pad)


def complex_grid(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_size_x: int,
    grid_size_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], grid_size_x)
    ys = np.linspace(ylim[0], ylim[1], grid_size_y)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return x_grid, y_grid, x_grid + 1j * y_grid


def block_nonnormal_singular_value_grid(
    eigenvalues: np.ndarray,
    pairs: np.ndarray,
    singles: np.ndarray,
    coupling: float,
    z_grid: np.ndarray,
) -> np.ndarray:
    """Compute sigma_min(zI-A) analytically over the block-diagonal matrix."""

    values = np.empty(z_grid.shape, dtype=float)
    pair_left = eigenvalues[pairs[:, 0]] if len(pairs) else np.empty(0, dtype=np.complex128)
    pair_right = eigenvalues[pairs[:, 1]] if len(pairs) else np.empty(0, dtype=np.complex128)
    single_values = eigenvalues[singles] if len(singles) else np.empty(0, dtype=np.complex128)
    coupling_squared = coupling * coupling

    chunk_rows = 20
    pair_batch = 64
    for row_start in range(0, z_grid.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, z_grid.shape[0])
        chunk = z_grid[row_start:row_stop, :]
        best = np.full(chunk.shape, np.inf, dtype=float)

        for block_start in range(0, len(pair_left), pair_batch):
            block_stop = min(block_start + pair_batch, len(pair_left))
            d1 = chunk[:, :, np.newaxis] - pair_left[np.newaxis, np.newaxis, block_start:block_stop]
            d2 = chunk[:, :, np.newaxis] - pair_right[np.newaxis, np.newaxis, block_start:block_stop]
            diagonal_1 = np.abs(d1) ** 2
            diagonal_2 = np.abs(d2) ** 2 + coupling_squared
            discriminant = np.maximum(
                (diagonal_1 - diagonal_2) ** 2 + 4.0 * coupling_squared * diagonal_1,
                0.0,
            )
            sigma_squared = 0.5 * (
                diagonal_1 + diagonal_2 - np.sqrt(discriminant)
            )
            best = np.minimum(
                best,
                np.sqrt(np.maximum(np.min(sigma_squared, axis=2), 0.0)),
            )

        for block_start in range(0, len(single_values), 256):
            block_stop = min(block_start + 256, len(single_values))
            sigma_single = np.min(
                np.abs(
                    chunk[:, :, np.newaxis]
                    - single_values[np.newaxis, np.newaxis, block_start:block_stop]
                ),
                axis=2,
            )
            best = np.minimum(best, sigma_single)

        values[row_start:row_stop, :] = best
        print(f"  sigma_min rows {row_start + 1}-{row_stop}/{z_grid.shape[0]}", flush=True)

    return values


def save_osu_contour_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sigma_min: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    levels: np.ndarray,
    colors: Tuple[str, ...],
    linewidth_range: Tuple[float, float],
    output_png: Path,
    dpi: int,
) -> None:
    """Render the name with OSU contours and no axes or colorbar."""

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    figure_height_inches = FIGURE_WIDTH_INCHES * y_span / x_span
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_INCHES, figure_height_inches),
        facecolor="white",
    )
    axis = figure.add_axes([0.01, 0.01, 0.98, 0.98])
    axis.set_facecolor("white")

    # Smallest levels are darkest; lighter gray contours recede outward.
    if len(colors) != len(levels):
        raise ValueError("Each contour level must have exactly one color.")
    linewidths = np.linspace(linewidth_range[0], linewidth_range[1], len(levels))
    axis.contour(
        x_grid,
        y_grid,
        sigma_min,
        levels=levels,
        colors=colors,
        linewidths=linewidths,
        zorder=1,
    )

    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_aspect(1.0, adjustable="box", anchor="C")
    axis.set_axis_off()

    # Resolve the transform now and fail loudly if a future layout change ever
    # stretches the data: one x-unit and one y-unit must occupy equal pixels.
    figure.canvas.draw()
    origin_pixels = axis.transData.transform((xlim[0], ylim[0]))
    x_unit_pixels = axis.transData.transform((xlim[0] + 1.0, ylim[0]))
    y_unit_pixels = axis.transData.transform((xlim[0], ylim[0] + 1.0))
    x_scale = float(np.linalg.norm(x_unit_pixels - origin_pixels))
    y_scale = float(np.linalg.norm(y_unit_pixels - origin_pixels))
    if not np.isclose(x_scale, y_scale, rtol=1.0e-10, atol=1.0e-10):
        raise RuntimeError(
            f"Unequal plotting scales detected: x={x_scale:.12g}px, y={y_scale:.12g}px"
        )

    save_kwargs = {
        "dpi": dpi,
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.06,
    }
    figure.savefig(output_png, **save_kwargs)
    plt.close(figure)


if __name__ == "__main__":
    main()

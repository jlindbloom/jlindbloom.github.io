"""Construct a matrix whose epsilon-pseudospectrum spells a name."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pseudospectra_matplotlib_config")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FLAT_DIR = SCRIPT_DIR / "flat"
OUTPUT_PDF = SCRIPT_DIR / "jonathan_lindbloom_pseudospectrum.pdf"
OUTPUT_PNG = SCRIPT_DIR / "jonathan_lindbloom_pseudospectrum.png"
OUTPUT_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_pseudospectrum_contour.pdf"
OUTPUT_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_pseudospectrum_contour.png"
OUTPUT_MULTI_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_pseudospectrum_multicolor_contours.pdf"
OUTPUT_MULTI_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_pseudospectrum_multicolor_contours.png"
OUTPUT_MATRIX = SCRIPT_DIR / "jonathan_lindbloom_matrix.npy"
OUTPUT_NONNORMAL_PDF = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum.pdf"
OUTPUT_NONNORMAL_PNG = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum.png"
OUTPUT_NONNORMAL_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_contour.pdf"
OUTPUT_NONNORMAL_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_contour.png"
OUTPUT_NONNORMAL_MULTI_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_multicolor_contours.pdf"
OUTPUT_NONNORMAL_MULTI_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_multicolor_contours.png"
OUTPUT_NONNORMAL_WIDE_MULTI_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_multicolor_contours_wide.pdf"
OUTPUT_NONNORMAL_WIDE_MULTI_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_multicolor_contours_wide.png"
OUTPUT_NONNORMAL_WIDER_MULTI_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_multicolor_contours_wider.pdf"
OUTPUT_NONNORMAL_WIDER_MULTI_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_multicolor_contours_wider.png"
OUTPUT_NONNORMAL_FULLSPAN_MULTI_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_multicolor_contours_fullspan.pdf"
OUTPUT_NONNORMAL_FULLSPAN_MULTI_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_pseudospectrum_multicolor_contours_fullspan.png"
OUTPUT_NONNORMAL_MATRIX = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_matrix.npy"
OUTPUT_ULTRANONNORMAL_PDF = SCRIPT_DIR / "jonathan_lindbloom_ultranonnormal_pseudospectrum.pdf"
OUTPUT_ULTRANONNORMAL_PNG = SCRIPT_DIR / "jonathan_lindbloom_ultranonnormal_pseudospectrum.png"
OUTPUT_ULTRANONNORMAL_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_ultranonnormal_pseudospectrum_contour.pdf"
OUTPUT_ULTRANONNORMAL_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_ultranonnormal_pseudospectrum_contour.png"
OUTPUT_ULTRANONNORMAL_MULTI_CONTOUR_PDF = SCRIPT_DIR / "jonathan_lindbloom_ultranonnormal_pseudospectrum_multicolor_contours.pdf"
OUTPUT_ULTRANONNORMAL_MULTI_CONTOUR_PNG = SCRIPT_DIR / "jonathan_lindbloom_ultranonnormal_pseudospectrum_multicolor_contours.png"
OUTPUT_ULTRANONNORMAL_MATRIX = SCRIPT_DIR / "jonathan_lindbloom_ultranonnormal_matrix.npy"
OUTPUT_FLAT_NONNORMAL_WIDER_MULTI_CONTOUR_PDF = FLAT_DIR / "jonathan_lindbloom_flat_nonnormal_pseudospectrum_multicolor_contours_wider.pdf"
OUTPUT_FLAT_NONNORMAL_WIDER_MULTI_CONTOUR_PNG = FLAT_DIR / "jonathan_lindbloom_flat_nonnormal_pseudospectrum_multicolor_contours_wider.png"
OUTPUT_FLAT_NONNORMAL_MATRIX = FLAT_DIR / "jonathan_lindbloom_flat_nonnormal_matrix.npy"
OUTPUT_NONNORMAL_WIDER_NUMERICAL_RANGE_PDF = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_numerical_range_wider.pdf"
OUTPUT_NONNORMAL_WIDER_NUMERICAL_RANGE_PNG = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_numerical_range_wider.png"
OUTPUT_NONNORMAL_WIDE_FILLED_NUMERICAL_RANGE_PDF = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_numerical_range_wide_filled.pdf"
OUTPUT_NONNORMAL_WIDE_FILLED_NUMERICAL_RANGE_PNG = SCRIPT_DIR / "jonathan_lindbloom_nonnormal_numerical_range_wide_filled.png"
OUTPUT_FLAT_NONNORMAL_WIDER_NUMERICAL_RANGE_PDF = FLAT_DIR / "jonathan_lindbloom_flat_nonnormal_numerical_range_wider.pdf"
OUTPUT_FLAT_NONNORMAL_WIDER_NUMERICAL_RANGE_PNG = FLAT_DIR / "jonathan_lindbloom_flat_nonnormal_numerical_range_wider.png"

LINE1 = "Jonathan"
LINE2 = "Lindbloom"
FLAT_LINE = "Jonathan Lindbloom"
FONT = FontProperties(family="DejaVu Sans", weight="bold")

TARGET_WIDTH = 14.0
LINE_GAP = 0.65
SAMPLE_PITCH = 0.12
EPSILON = 0.092
GRID_SIZE_X = 950
GRID_SIZE_Y = 560
NONNORMAL_COUPLING = 0.34
ULTRANONNORMAL_COUPLING_1 = 0.56
ULTRANONNORMAL_COUPLING_2 = 0.28

INK = "#20242a"
FILL = "#dbe7f4"
RULE = "#d7dbe3"
NUM_RANGE = "#9fd0ee"
EIGEN_TICK = "#9aa0a6"
NUM_RANGE_EDGE = "#2c6ea6"


def main() -> None:
    FLAT_DIR.mkdir(parents=True, exist_ok=True)

    path = build_centered_name_path()
    sample_points = sample_points_in_path(path)
    matrix = np.diag(sample_points.astype(np.complex128))
    np.save(OUTPUT_MATRIX, matrix)
    print(f"wrote {OUTPUT_MATRIX}", flush=True)

    xlim, ylim = plotting_limits(path)
    x_grid, y_grid, z_grid = complex_grid(xlim, ylim)
    sigma_min = diagonal_singular_value_grid(sample_points, z_grid)
    save_plot(path, sample_points, x_grid, y_grid, sigma_min, xlim, ylim, OUTPUT_PDF, OUTPUT_PNG)
    save_contour_plot(path, x_grid, y_grid, sigma_min, xlim, ylim, OUTPUT_CONTOUR_PDF, OUTPUT_CONTOUR_PNG)
    save_multicolor_contour_plot(
        path,
        x_grid,
        y_grid,
        sigma_min,
        xlim,
        ylim,
        OUTPUT_MULTI_CONTOUR_PDF,
        OUTPUT_MULTI_CONTOUR_PNG,
    )

    pairs, singles = build_local_pairs(sample_points)
    nonnormal_matrix = build_nonnormal_matrix(sample_points, pairs, singles, NONNORMAL_COUPLING)
    np.save(OUTPUT_NONNORMAL_MATRIX, nonnormal_matrix)
    print(f"wrote {OUTPUT_NONNORMAL_MATRIX}", flush=True)

    sigma_min_nonnormal = block_nonnormal_singular_value_grid(
        sample_points,
        pairs,
        singles,
        NONNORMAL_COUPLING,
        z_grid,
    )
    save_plot(
        path,
        sample_points,
        x_grid,
        y_grid,
        sigma_min_nonnormal,
        xlim,
        ylim,
        OUTPUT_NONNORMAL_PDF,
        OUTPUT_NONNORMAL_PNG,
    )
    save_contour_plot(
        path,
        x_grid,
        y_grid,
        sigma_min_nonnormal,
        xlim,
        ylim,
        OUTPUT_NONNORMAL_CONTOUR_PDF,
        OUTPUT_NONNORMAL_CONTOUR_PNG,
    )
    save_multicolor_contour_plot(
        path,
        x_grid,
        y_grid,
        sigma_min_nonnormal,
        xlim,
        ylim,
        OUTPUT_NONNORMAL_MULTI_CONTOUR_PDF,
        OUTPUT_NONNORMAL_MULTI_CONTOUR_PNG,
    )
    wide_levels = EPSILON * np.array([0.35, 0.5, 0.7, 0.9, 1.0, 1.4, 1.9, 2.6, 3.5, 4.8, 6.5, 8.5, 11.0])
    save_multicolor_contour_plot(
        path,
        x_grid,
        y_grid,
        sigma_min_nonnormal,
        xlim,
        ylim,
        OUTPUT_NONNORMAL_WIDE_MULTI_CONTOUR_PDF,
        OUTPUT_NONNORMAL_WIDE_MULTI_CONTOUR_PNG,
        levels=wide_levels,
    )
    save_filled_numerical_range_plot(
        sample_points,
        pairs,
        singles,
        OUTPUT_NONNORMAL_WIDE_FILLED_NUMERICAL_RANGE_PDF,
        OUTPUT_NONNORMAL_WIDE_FILLED_NUMERICAL_RANGE_PNG,
        coupling=NONNORMAL_COUPLING,
    )
    wider_levels = EPSILON * np.array([0.35, 0.5, 0.7, 0.9, 1.0, 1.5, 2.1, 3.0, 4.2, 5.8, 8.0, 11.0, 15.0, 20.0])
    wider_xlim, wider_ylim = plotting_limits(path, contour_levels=wider_levels, padding_scale=1.28)
    wider_x_grid, wider_y_grid, wider_z_grid = complex_grid(wider_xlim, wider_ylim)
    sigma_min_nonnormal_wider = block_nonnormal_singular_value_grid(
        sample_points,
        pairs,
        singles,
        NONNORMAL_COUPLING,
        wider_z_grid,
    )
    save_multicolor_contour_plot(
        path,
        wider_x_grid,
        wider_y_grid,
        sigma_min_nonnormal_wider,
        wider_xlim,
        wider_ylim,
        OUTPUT_NONNORMAL_WIDER_MULTI_CONTOUR_PDF,
        OUTPUT_NONNORMAL_WIDER_MULTI_CONTOUR_PNG,
        levels=wider_levels,
    )
    save_numerical_range_plot(
        sample_points,
        pairs,
        singles,
        wider_xlim,
        wider_ylim,
        OUTPUT_NONNORMAL_WIDER_NUMERICAL_RANGE_PDF,
        OUTPUT_NONNORMAL_WIDER_NUMERICAL_RANGE_PNG,
        coupling=NONNORMAL_COUPLING,
    )
    fullspan_levels = EPSILON * np.array(
        [
            0.12,
            0.18,
            0.26,
            0.35,
            0.5,
            0.7,
            0.9,
            1.0,
            1.3,
            1.7,
            2.3,
            3.0,
            4.0,
            5.4,
            7.2,
            9.5,
            12.5,
            16.0,
            20.0,
        ]
    )
    save_multicolor_contour_plot(
        path,
        wider_x_grid,
        wider_y_grid,
        sigma_min_nonnormal_wider,
        wider_xlim,
        wider_ylim,
        OUTPUT_NONNORMAL_FULLSPAN_MULTI_CONTOUR_PDF,
        OUTPUT_NONNORMAL_FULLSPAN_MULTI_CONTOUR_PNG,
        levels=fullspan_levels,
    )

    strong_groups = build_local_groups(sample_points, group_size=3)
    ultranonnormal_matrix = build_grouped_nonnormal_matrix(
        sample_points,
        strong_groups,
        ULTRANONNORMAL_COUPLING_1,
        ULTRANONNORMAL_COUPLING_2,
    )
    np.save(OUTPUT_ULTRANONNORMAL_MATRIX, ultranonnormal_matrix)
    print(f"wrote {OUTPUT_ULTRANONNORMAL_MATRIX}", flush=True)

    sigma_min_ultranonnormal = grouped_nonnormal_singular_value_grid(
        sample_points,
        strong_groups,
        ULTRANONNORMAL_COUPLING_1,
        ULTRANONNORMAL_COUPLING_2,
        z_grid,
    )
    save_plot(
        path,
        sample_points,
        x_grid,
        y_grid,
        sigma_min_ultranonnormal,
        xlim,
        ylim,
        OUTPUT_ULTRANONNORMAL_PDF,
        OUTPUT_ULTRANONNORMAL_PNG,
    )
    save_contour_plot(
        path,
        x_grid,
        y_grid,
        sigma_min_ultranonnormal,
        xlim,
        ylim,
        OUTPUT_ULTRANONNORMAL_CONTOUR_PDF,
        OUTPUT_ULTRANONNORMAL_CONTOUR_PNG,
    )
    save_multicolor_contour_plot(
        path,
        x_grid,
        y_grid,
        sigma_min_ultranonnormal,
        xlim,
        ylim,
        OUTPUT_ULTRANONNORMAL_MULTI_CONTOUR_PDF,
        OUTPUT_ULTRANONNORMAL_MULTI_CONTOUR_PNG,
    )

    flat_path = build_flat_name_path()
    flat_sample_points = sample_points_in_path(flat_path)
    flat_pairs, flat_singles = build_local_pairs(flat_sample_points)
    flat_nonnormal_matrix = build_nonnormal_matrix(flat_sample_points, flat_pairs, flat_singles, NONNORMAL_COUPLING)
    np.save(OUTPUT_FLAT_NONNORMAL_MATRIX, flat_nonnormal_matrix)
    print(f"wrote {OUTPUT_FLAT_NONNORMAL_MATRIX}", flush=True)

    flat_levels = EPSILON * np.array([0.35, 0.5, 0.7, 0.9, 1.0, 1.5, 2.1, 3.0, 4.2, 5.8, 7.6, 9.6, 12.0, 14.5])
    flat_xlim, flat_ylim = plotting_limits(flat_path, contour_levels=flat_levels, padding_scale=1.18)
    flat_x_grid, flat_y_grid, flat_z_grid = complex_grid(flat_xlim, flat_ylim)
    flat_sigma_min_nonnormal = block_nonnormal_singular_value_grid(
        flat_sample_points,
        flat_pairs,
        flat_singles,
        NONNORMAL_COUPLING,
        flat_z_grid,
    )
    save_multicolor_contour_plot(
        flat_path,
        flat_x_grid,
        flat_y_grid,
        flat_sigma_min_nonnormal,
        flat_xlim,
        flat_ylim,
        OUTPUT_FLAT_NONNORMAL_WIDER_MULTI_CONTOUR_PDF,
        OUTPUT_FLAT_NONNORMAL_WIDER_MULTI_CONTOUR_PNG,
        levels=flat_levels,
        show_axes=False,
    )
    save_numerical_range_plot(
        flat_sample_points,
        flat_pairs,
        flat_singles,
        flat_xlim,
        flat_ylim,
        OUTPUT_FLAT_NONNORMAL_WIDER_NUMERICAL_RANGE_PDF,
        OUTPUT_FLAT_NONNORMAL_WIDER_NUMERICAL_RANGE_PNG,
        coupling=NONNORMAL_COUPLING,
        show_axes=False,
        show_eigen_ticks=False,
    )


def build_centered_name_path() -> MplPath:
    line1 = TextPath((0.0, 0.0), LINE1, size=1.0, prop=FONT)
    line2 = TextPath((0.0, 0.0), LINE2, size=1.0, prop=FONT)

    ext1 = line1.get_extents()
    ext2 = line2.get_extents()

    line1_centered = line1.transformed(Affine2D().translate(-0.5 * (ext1.x0 + ext1.x1), 0.0))
    line2_centered = line2.transformed(Affine2D().translate(-0.5 * (ext2.x0 + ext2.x1), 0.0))

    ext1c = line1_centered.get_extents()
    ext2c = line2_centered.get_extents()
    line1_height = ext1c.y1 - ext1c.y0
    line2_height = ext2c.y1 - ext2c.y0

    line1_final = line1_centered.transformed(Affine2D().translate(0.0, 0.5 * (line2_height + LINE_GAP)))
    line2_final = line2_centered.transformed(Affine2D().translate(0.0, -0.5 * (line1_height + LINE_GAP)))

    vertices = np.concatenate([line1_final.vertices, line2_final.vertices], axis=0)
    codes = np.concatenate([line1_final.codes, line2_final.codes], axis=0)
    combined = MplPath(vertices, codes)

    ext = combined.get_extents()
    scale = TARGET_WIDTH / (ext.x1 - ext.x0)
    combined = combined.transformed(Affine2D().scale(scale, scale))
    ext = combined.get_extents()
    combined = combined.transformed(Affine2D().translate(-0.5 * (ext.x0 + ext.x1), -0.5 * (ext.y0 + ext.y1)))
    return combined


def build_flat_name_path() -> MplPath:
    line = TextPath((0.0, 0.0), FLAT_LINE, size=1.0, prop=FONT)
    ext = line.get_extents()
    centered = line.transformed(Affine2D().translate(-0.5 * (ext.x0 + ext.x1), -0.5 * (ext.y0 + ext.y1)))
    ext = centered.get_extents()
    scale = TARGET_WIDTH / (ext.x1 - ext.x0)
    centered = centered.transformed(Affine2D().scale(scale, scale))
    ext = centered.get_extents()
    centered = centered.transformed(Affine2D().translate(-0.5 * (ext.x0 + ext.x1), -0.5 * (ext.y0 + ext.y1)))
    return centered


def sample_points_in_path(path: MplPath) -> np.ndarray:
    ext = path.get_extents()
    xs = np.arange(ext.x0, ext.x1 + SAMPLE_PITCH, SAMPLE_PITCH)
    ys = np.arange(ext.y0, ext.y1 + SAMPLE_PITCH, SAMPLE_PITCH)
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    # Use an even-odd fill test over the individual glyph polygons so counters
    # in letters like o, a, b, and d remain empty.
    inside = np.zeros(len(points), dtype=bool)
    for polygon in path.to_polygons(closed_only=True):
        if len(polygon) < 3:
            continue
        polygon_path = MplPath(polygon, closed=True)
        inside ^= polygon_path.contains_points(points, radius=0.0)

    selected = points[inside]
    return selected[:, 0] + 1j * selected[:, 1]


def plotting_limits(
    path: MplPath,
    contour_levels: np.ndarray | None = None,
    padding_scale: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    ext = path.get_extents()
    x_margin = 1.1
    y_margin = 0.9
    if contour_levels is not None and len(contour_levels) > 0:
        level_radius = float(np.max(contour_levels)) / EPSILON * SAMPLE_PITCH * 0.62
        x_margin += padding_scale * level_radius
        y_margin += padding_scale * level_radius
    x_radius = max(abs(ext.x0), abs(ext.x1)) + x_margin
    y_radius = max(abs(ext.y0), abs(ext.y1)) + y_margin
    return (-x_radius, x_radius), (-y_radius, y_radius)


def complex_grid(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], GRID_SIZE_X)
    ys = np.linspace(ylim[0], ylim[1], GRID_SIZE_Y)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return x_grid, y_grid, x_grid + 1j * y_grid


def diagonal_singular_value_grid(eigenvalues: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    values = np.empty(z_grid.shape, dtype=float)
    chunk_rows = 56
    for row_start in range(0, z_grid.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, z_grid.shape[0])
        chunk = z_grid[row_start:row_stop, :, np.newaxis]
        values[row_start:row_stop, :] = np.min(np.abs(chunk - eigenvalues[np.newaxis, np.newaxis, :]), axis=2)
        print(f"  rows {row_start + 1}-{row_stop}/{z_grid.shape[0]}", flush=True)
    return values


def build_local_pairs(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.column_stack([eigenvalues.real, eigenvalues.imag])
    used = np.zeros(len(eigenvalues), dtype=bool)
    pair_list: list[tuple[int, int]] = []
    leftovers: list[int] = []

    unique_y = np.unique(points[:, 1])
    for y in unique_y:
        row_indices = np.flatnonzero(points[:, 1] == y)
        row_indices = row_indices[np.argsort(points[row_indices, 0])]
        stop = len(row_indices) - (len(row_indices) % 2)
        for k in range(0, stop, 2):
            i = int(row_indices[k])
            j = int(row_indices[k + 1])
            pair_list.append((i, j))
            used[i] = True
            used[j] = True
        if len(row_indices) % 2 == 1:
            leftovers.append(int(row_indices[-1]))

    if leftovers:
        leftover_indices = np.array(leftovers, dtype=int)
        leftover_indices = leftover_indices[np.argsort(points[leftover_indices, 0], kind="mergesort")]
        grouped_leftovers: list[int] = []
        for x in np.unique(points[leftover_indices, 0]):
            column_indices = leftover_indices[points[leftover_indices, 0] == x]
            column_indices = column_indices[np.argsort(points[column_indices, 1])]
            stop = len(column_indices) - (len(column_indices) % 2)
            for k in range(0, stop, 2):
                i = int(column_indices[k])
                j = int(column_indices[k + 1])
                pair_list.append((i, j))
                used[i] = True
                used[j] = True
            if len(column_indices) % 2 == 1:
                grouped_leftovers.append(int(column_indices[-1]))
        leftovers = grouped_leftovers

    singles = np.flatnonzero(~used)
    if leftovers:
        leftovers_array = np.array(leftovers, dtype=int)
        leftovers_array = leftovers_array[~used[leftovers_array]]
        if len(leftovers_array) > 0:
            singles = np.unique(np.concatenate([singles, leftovers_array]))

    return np.array(pair_list, dtype=int), singles.astype(int)


def build_local_groups(eigenvalues: np.ndarray, group_size: int) -> list[np.ndarray]:
    points = np.column_stack([eigenvalues.real, eigenvalues.imag])
    used = np.zeros(len(eigenvalues), dtype=bool)
    groups: list[np.ndarray] = []
    leftovers: list[int] = []

    unique_y = np.unique(points[:, 1])
    for y in unique_y:
        row_indices = np.flatnonzero(points[:, 1] == y)
        row_indices = row_indices[np.argsort(points[row_indices, 0])]
        stop = len(row_indices) - (len(row_indices) % group_size)
        for k in range(0, stop, group_size):
            group = row_indices[k : k + group_size].astype(int)
            groups.append(group)
            used[group] = True
        leftovers.extend(int(idx) for idx in row_indices[stop:])

    if leftovers:
        leftover_indices = np.array(leftovers, dtype=int)
        leftover_indices = leftover_indices[np.argsort(points[leftover_indices, 0], kind="mergesort")]
        regrouped_leftovers: list[int] = []
        for x in np.unique(points[leftover_indices, 0]):
            column_indices = leftover_indices[points[leftover_indices, 0] == x]
            column_indices = column_indices[np.argsort(points[column_indices, 1])]
            stop = len(column_indices) - (len(column_indices) % group_size)
            for k in range(0, stop, group_size):
                group = column_indices[k : k + group_size].astype(int)
                groups.append(group)
                used[group] = True
            regrouped_leftovers.extend(int(idx) for idx in column_indices[stop:])
        leftovers = regrouped_leftovers

    if leftovers:
        leftover_indices = np.array([idx for idx in leftovers if not used[idx]], dtype=int)
        if len(leftover_indices) >= 2:
            leftover_indices = leftover_indices[np.argsort(points[leftover_indices, 0] + 1j * points[leftover_indices, 1], kind="mergesort")]
            stop = len(leftover_indices) - (len(leftover_indices) % 2)
            for k in range(0, stop, 2):
                group = leftover_indices[k : k + 2].astype(int)
                groups.append(group)
                used[group] = True
            leftovers = [int(idx) for idx in leftover_indices[stop:] if not used[idx]]

    singles = np.flatnonzero(~used)
    for idx in singles:
        groups.append(np.array([int(idx)], dtype=int))

    return groups


def build_nonnormal_matrix(
    eigenvalues: np.ndarray,
    pairs: np.ndarray,
    singles: np.ndarray,
    coupling: float,
) -> np.ndarray:
    n = len(eigenvalues)
    matrix = np.zeros((n, n), dtype=np.complex128)
    cursor = 0
    for i, j in pairs:
        matrix[cursor, cursor] = eigenvalues[i]
        matrix[cursor, cursor + 1] = coupling
        matrix[cursor + 1, cursor + 1] = eigenvalues[j]
        cursor += 2
    for idx in singles:
        matrix[cursor, cursor] = eigenvalues[idx]
        cursor += 1
    return matrix


def build_grouped_nonnormal_matrix(
    eigenvalues: np.ndarray,
    groups: list[np.ndarray],
    coupling_1: float,
    coupling_2: float,
) -> np.ndarray:
    n = len(eigenvalues)
    matrix = np.zeros((n, n), dtype=np.complex128)
    cursor = 0
    for group in groups:
        size = len(group)
        values = eigenvalues[group]
        matrix[cursor : cursor + size, cursor : cursor + size] = np.diag(values)
        if size >= 2:
            matrix[cursor, cursor + 1] = coupling_1
        if size >= 3:
            matrix[cursor + 1, cursor + 2] = coupling_1
            matrix[cursor, cursor + 2] = coupling_2
        cursor += size
    return matrix


def block_nonnormal_singular_value_grid(
    eigenvalues: np.ndarray,
    pairs: np.ndarray,
    singles: np.ndarray,
    coupling: float,
    z_grid: np.ndarray,
) -> np.ndarray:
    values = np.empty(z_grid.shape, dtype=float)
    chunk_rows = 20
    block_batch = 64
    coupling_sq = coupling * coupling

    pair_left = eigenvalues[pairs[:, 0]] if len(pairs) else np.empty(0, dtype=np.complex128)
    pair_right = eigenvalues[pairs[:, 1]] if len(pairs) else np.empty(0, dtype=np.complex128)
    single_values = eigenvalues[singles] if len(singles) else np.empty(0, dtype=np.complex128)

    for row_start in range(0, z_grid.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, z_grid.shape[0])
        chunk = z_grid[row_start:row_stop, :]
        best = np.full(chunk.shape, np.inf, dtype=float)

        for block_start in range(0, len(pair_left), block_batch):
            block_stop = min(block_start + block_batch, len(pair_left))
            d1 = chunk[:, :, np.newaxis] - pair_left[np.newaxis, np.newaxis, block_start:block_stop]
            d2 = chunk[:, :, np.newaxis] - pair_right[np.newaxis, np.newaxis, block_start:block_stop]
            a = np.abs(d1) ** 2
            c = np.abs(d2) ** 2 + coupling_sq
            disc = np.maximum((a - c) ** 2 + 4.0 * coupling_sq * a, 0.0)
            sigma_sq = 0.5 * (a + c - np.sqrt(disc))
            best = np.minimum(best, np.sqrt(np.maximum(np.min(sigma_sq, axis=2), 0.0)))

        for block_start in range(0, len(single_values), 256):
            block_stop = min(block_start + 256, len(single_values))
            sigma_single = np.min(
                np.abs(chunk[:, :, np.newaxis] - single_values[np.newaxis, np.newaxis, block_start:block_stop]),
                axis=2,
            )
            best = np.minimum(best, sigma_single)

        values[row_start:row_stop, :] = best
        print(f"  nonnormal rows {row_start + 1}-{row_stop}/{z_grid.shape[0]}", flush=True)

    return values


def grouped_nonnormal_singular_value_grid(
    eigenvalues: np.ndarray,
    groups: list[np.ndarray],
    coupling_1: float,
    coupling_2: float,
    z_grid: np.ndarray,
) -> np.ndarray:
    values = np.empty(z_grid.shape, dtype=float)
    chunk_rows = 8

    groups_by_size: dict[int, list[np.ndarray]] = {}
    for group in groups:
        groups_by_size.setdefault(len(group), []).append(group)

    for row_start in range(0, z_grid.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, z_grid.shape[0])
        chunk = z_grid[row_start:row_stop, :]
        flat_chunk = chunk.reshape(-1)
        best = np.full(flat_chunk.shape, np.inf, dtype=float)

        if 1 in groups_by_size:
            single_values = eigenvalues[np.array([group[0] for group in groups_by_size[1]], dtype=int)]
            for block_start in range(0, len(single_values), 256):
                block_stop = min(block_start + 256, len(single_values))
                sigma_single = np.min(
                    np.abs(flat_chunk[:, np.newaxis] - single_values[np.newaxis, block_start:block_stop]),
                    axis=1,
                )
                best = np.minimum(best, sigma_single)

        if 2 in groups_by_size:
            pair_groups = np.array(groups_by_size[2], dtype=int)
            pair_left = eigenvalues[pair_groups[:, 0]]
            pair_right = eigenvalues[pair_groups[:, 1]]
            coupling_sq = coupling_1 * coupling_1
            for block_start in range(0, len(pair_left), 64):
                block_stop = min(block_start + 64, len(pair_left))
                d1 = flat_chunk[:, np.newaxis] - pair_left[np.newaxis, block_start:block_stop]
                d2 = flat_chunk[:, np.newaxis] - pair_right[np.newaxis, block_start:block_stop]
                a = np.abs(d1) ** 2
                c = np.abs(d2) ** 2 + coupling_sq
                disc = np.maximum((a - c) ** 2 + 4.0 * coupling_sq * a, 0.0)
                sigma_sq = 0.5 * (a + c - np.sqrt(disc))
                best = np.minimum(best, np.sqrt(np.maximum(np.min(sigma_sq, axis=1), 0.0)))

        if 3 in groups_by_size:
            triple_groups = np.array(groups_by_size[3], dtype=int)
            triple_values = eigenvalues[triple_groups]
            base = np.zeros((len(triple_groups), 3, 3), dtype=np.complex128)
            base[:, 0, 1] = coupling_1
            base[:, 1, 2] = coupling_1
            base[:, 0, 2] = coupling_2

            z_batch = 512
            group_batch = 24
            eye3 = np.eye(3, dtype=np.complex128)
            for z_start in range(0, len(flat_chunk), z_batch):
                z_stop = min(z_start + z_batch, len(flat_chunk))
                z_values = flat_chunk[z_start:z_stop]
                local_best = np.full(len(z_values), np.inf, dtype=float)
                for group_start in range(0, len(triple_groups), group_batch):
                    group_stop = min(group_start + group_batch, len(triple_groups))
                    block = base[group_start:group_stop].copy()
                    block[:, 0, 0] = triple_values[group_start:group_stop, 0]
                    block[:, 1, 1] = triple_values[group_start:group_stop, 1]
                    block[:, 2, 2] = triple_values[group_start:group_stop, 2]
                    matrices = (
                        z_values[:, np.newaxis, np.newaxis, np.newaxis] * eye3[np.newaxis, np.newaxis, :, :]
                        - block[np.newaxis, :, :, :]
                    )
                    sigma = np.linalg.svd(matrices, compute_uv=False)
                    local_best = np.minimum(local_best, np.min(sigma[:, :, -1], axis=1))
                best[z_start:z_stop] = np.minimum(best[z_start:z_stop], local_best)

        values[row_start:row_stop, :] = best.reshape(chunk.shape)
        print(f"  ultranonnormal rows {row_start + 1}-{row_stop}/{z_grid.shape[0]}", flush=True)

    return values


def numerical_range_boundary(
    eigenvalues: np.ndarray,
    pairs: np.ndarray,
    singles: np.ndarray,
    coupling: float,
    num_angles: int = 960,
) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False)
    boundary = np.empty(num_angles, dtype=np.complex128)

    pair_left = eigenvalues[pairs[:, 0]] if len(pairs) else np.empty(0, dtype=np.complex128)
    pair_right = eigenvalues[pairs[:, 1]] if len(pairs) else np.empty(0, dtype=np.complex128)
    single_values = eigenvalues[singles] if len(singles) else np.empty(0, dtype=np.complex128)

    for k, theta in enumerate(thetas):
        phase = np.exp(-1j * theta)
        best_support = -np.inf
        best_point = 0.0 + 0.0j

        if len(single_values):
            rotated = np.real(phase * single_values)
            idx = int(np.argmax(rotated))
            best_support = float(rotated[idx])
            best_point = single_values[idx]

        if len(pair_left):
            alpha = np.real(phase * pair_left)
            delta = np.real(phase * pair_right)
            beta = 0.5 * coupling * phase
            gap = 0.5 * (alpha - delta)
            rad = np.sqrt(gap * gap + np.abs(beta) ** 2)
            lambdas = 0.5 * (alpha + delta) + rad
            idx = int(np.argmax(lambdas))
            if float(lambdas[idx]) > best_support:
                if rad[idx] > 1.0e-14:
                    x = beta
                    y = lambdas[idx] - alpha[idx]
                    norm = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2)
                    v1 = x / norm
                    v2 = y / norm
                else:
                    v1 = 1.0 + 0.0j
                    v2 = 0.0 + 0.0j
                lam1 = pair_left[idx]
                lam2 = pair_right[idx]
                best_point = np.conj(v1) * lam1 * v1 + np.conj(v1) * coupling * v2 + np.conj(v2) * lam2 * v2
                best_support = float(lambdas[idx])

        boundary[k] = best_point

    return boundary


def save_numerical_range_plot(
    eigenvalues: np.ndarray,
    pairs: np.ndarray,
    singles: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    output_pdf: Path,
    output_png: Path,
    coupling: float = NONNORMAL_COUPLING,
    show_axes: bool = True,
    show_eigen_ticks: bool = True,
) -> None:
    boundary = numerical_range_boundary(eigenvalues, pairs, singles, coupling)
    tick_half_height = 0.055
    boundary_xlim, boundary_ylim = boundary_plotting_limits(boundary)

    fig, ax = plt.subplots(figsize=(14.0, 8.0), constrained_layout=True)
    ax.set_facecolor("white")
    if show_axes:
        ax.axhline(0.0, color=RULE, linewidth=0.7, zorder=0)
        ax.axvline(0.0, color=RULE, linewidth=0.7, zorder=0)

    ax.plot(boundary.real, boundary.imag, color=NUM_RANGE, linewidth=2.0, zorder=1)
    ax.plot(
        [boundary[-1].real, boundary[0].real],
        [boundary[-1].imag, boundary[0].imag],
        color=NUM_RANGE,
        linewidth=2.0,
        zorder=1,
    )
    if show_eigen_ticks:
        ax.vlines(
            eigenvalues.real,
            eigenvalues.imag - tick_half_height,
            eigenvalues.imag + tick_half_height,
            color=EIGEN_TICK,
            linewidth=0.5,
            alpha=0.65,
            zorder=2,
        )

    ax.set_xlim(*boundary_xlim)
    ax.set_ylim(*boundary_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_pdf, dpi=300, facecolor="white")
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {output_pdf}", flush=True)
    print(f"wrote {output_png}", flush=True)


def save_filled_numerical_range_plot(
    eigenvalues: np.ndarray,
    pairs: np.ndarray,
    singles: np.ndarray,
    output_pdf: Path,
    output_png: Path,
    coupling: float = NONNORMAL_COUPLING,
    show_axes: bool = True,
) -> None:
    boundary = numerical_range_boundary(eigenvalues, pairs, singles, coupling)
    boundary_xlim, boundary_ylim = boundary_plotting_limits(boundary)

    fig, ax = plt.subplots(figsize=(14.0, 8.0), constrained_layout=True)
    ax.set_facecolor("white")
    if show_axes:
        ax.axhline(0.0, color=RULE, linewidth=0.7, zorder=0)
        ax.axvline(0.0, color=RULE, linewidth=0.7, zorder=0)

    ax.fill(boundary.real, boundary.imag, facecolor=NUM_RANGE, alpha=0.32, edgecolor="none", zorder=1)
    ax.plot(boundary.real, boundary.imag, color=NUM_RANGE_EDGE, linewidth=2.0, zorder=2)
    ax.plot(
        [boundary[-1].real, boundary[0].real],
        [boundary[-1].imag, boundary[0].imag],
        color=NUM_RANGE_EDGE,
        linewidth=2.0,
        zorder=2,
    )
    ax.scatter(
        eigenvalues.real,
        eigenvalues.imag,
        marker="x",
        s=10.0,
        color=EIGEN_TICK,
        alpha=0.45,
        linewidths=0.55,
        zorder=3,
    )

    ax.set_xlim(*boundary_xlim)
    ax.set_ylim(*boundary_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_pdf, dpi=300, facecolor="white")
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {output_pdf}", flush=True)
    print(f"wrote {output_png}", flush=True)


def boundary_plotting_limits(boundary: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    real_min = float(np.min(boundary.real))
    real_max = float(np.max(boundary.real))
    imag_min = float(np.min(boundary.imag))
    imag_max = float(np.max(boundary.imag))
    x_pad = max(0.24, 0.08 * (real_max - real_min))
    y_pad = max(0.18, 0.18 * max(imag_max - imag_min, 0.25))
    return (real_min - x_pad, real_max + x_pad), (imag_min - y_pad, imag_max + y_pad)


def save_plot(
    path: MplPath,
    eigenvalues: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sigma_min: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    output_pdf: Path,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 8.0), constrained_layout=True)
    ax.set_facecolor("white")
    ax.axhline(0.0, color=RULE, linewidth=0.7, zorder=0)
    ax.axvline(0.0, color=RULE, linewidth=0.7, zorder=0)

    ax.contourf(
        x_grid,
        y_grid,
        sigma_min,
        levels=[0.0, EPSILON],
        colors=[FILL],
        alpha=0.95,
        zorder=1,
    )
    ax.contour(
        x_grid,
        y_grid,
        sigma_min,
        levels=[EPSILON],
        colors=[INK],
        linewidths=1.1,
        zorder=2,
    )

    ax.scatter(eigenvalues.real, eigenvalues.imag, s=2.0, color="#6b7280", alpha=0.25, linewidths=0, zorder=3)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_pdf, dpi=300, facecolor="white")
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {output_pdf}", flush=True)
    print(f"wrote {output_png}", flush=True)


def save_contour_plot(
    path: MplPath,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sigma_min: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    output_pdf: Path,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 8.0), constrained_layout=True)
    ax.set_facecolor("white")
    ax.axhline(0.0, color=RULE, linewidth=0.7, zorder=0)
    ax.axvline(0.0, color=RULE, linewidth=0.7, zorder=0)

    ax.contour(
        x_grid,
        y_grid,
        sigma_min,
        levels=[EPSILON],
        colors=[INK],
        linewidths=1.5,
        zorder=1,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_pdf, dpi=300, facecolor="white")
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {output_pdf}", flush=True)
    print(f"wrote {output_png}", flush=True)


def save_multicolor_contour_plot(
    path: MplPath,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sigma_min: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    output_pdf: Path,
    output_png: Path,
    levels: np.ndarray | None = None,
    show_axes: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(14.0, 8.0), constrained_layout=True)
    ax.set_facecolor("white")
    if show_axes:
        ax.axhline(0.0, color=RULE, linewidth=0.7, zorder=0)
        ax.axvline(0.0, color=RULE, linewidth=0.7, zorder=0)

    if levels is None:
        levels = EPSILON * np.array([0.35, 0.5, 0.7, 0.9, 1.0, 1.3, 1.7, 2.2, 2.9, 3.8, 5.0])
    ax.contour(
        x_grid,
        y_grid,
        sigma_min,
        levels=levels,
        cmap="YlGnBu_r",
        linewidths=1.35,
        zorder=1,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_pdf, dpi=300, facecolor="white")
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {output_pdf}", flush=True)
    print(f"wrote {output_png}", flush=True)


if __name__ == "__main__":
    main()

"""Create a print-ready US Letter PDF matching the supplied Plain TeX nameplate.

The page contains one 18.8 cm by 3 cm black plate, horizontally centered and
positioned one inch from the top edge. The 12-level, equal-scale contour art is
drawn as vectors with a light-to-scarlet palette designed for a black ground.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm, inch
from reportlab.pdfgen.canvas import Canvas

import make_jonathan_lindbloom_inline_osu_pseudospectrum as art


SCRIPT_DIR = Path(__file__).resolve().parent
SITE_DIR = SCRIPT_DIR.parents[2]
OUTPUT_PDF = SITE_DIR / "output" / "pdf" / "jonathan-lindbloom-pseudospectrum-nameplate-letter.pdf"

PAGE_WIDTH, PAGE_HEIGHT = letter
PLATE_WIDTH = 18.8 * cm
PLATE_HEIGHT = 3.0 * cm
PLATE_X = 0.5 * (PAGE_WIDTH - PLATE_WIDTH)
PLATE_Y = PAGE_HEIGHT - inch - PLATE_HEIGHT

# The supplied template centers its letters inside a 15 cm measure. The
# pseudospectral halo extends beyond the letterforms, so allow the full contour
# art up to 18 cm; this leaves the letterforms close to that original 15 cm span.
ART_MAX_WIDTH = 18.0 * cm
ART_MAX_HEIGHT = 2.70 * cm

LEVEL_FACTORS = art.MEDIUM_INNER_LEVEL_FACTORS
LINEWIDTHS = np.linspace(0.56, 1.46, len(LEVEL_FACTORS))

# Smallest epsilon to largest.  Keep a bright inner highlight, then place
# Ohio State scarlet on the small-to-medium levels that trace the letters.
# The wider pseudospectral halo recedes through neutral grays into black.
DARK_GROUND_CONTOUR_COLORS = (
    "#FFFFFF",
    "#EFF1F2",
    "#BA0C2F",
    "#BA0C2F",
    "#BA0C2F",
    "#BA0C2F",
    "#BFC6CB",
    "#A7B1B7",
    "#868E92",
    "#646A6E",
    "#3F4443",
    "#212325",
)


def main() -> None:
    levels = art.BASE_EPSILON * LEVEL_FACTORS
    contour_groups = build_contour_paths(levels)
    bounds = contour_bounds(contour_groups)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    render_pdf(OUTPUT_PDF, contour_groups, bounds)

    print(f"page: {PAGE_WIDTH:.6f} x {PAGE_HEIGHT:.6f} pt (US Letter)")
    print(
        "plate: "
        f"{PLATE_WIDTH:.6f} x {PLATE_HEIGHT:.6f} pt "
        f"at ({PLATE_X:.6f}, {PLATE_Y:.6f})"
    )
    print(f"wrote {OUTPUT_PDF}")


def build_contour_paths(levels: np.ndarray) -> List[List[MplPath]]:
    """Compute the medium-density contour paths in numerical coordinates."""

    name_path = art.build_inline_name_path()
    eigenvalues = art.sample_points_in_path(name_path)
    pairs, singles = art.build_local_pairs(eigenvalues)

    widest_levels = art.BASE_EPSILON * art.DENSE_INNER_LEVEL_FACTORS
    xlim, ylim = art.plotting_limits(name_path, widest_levels)
    x_grid, y_grid, z_grid = art.complex_grid(
        xlim,
        ylim,
        art.DEFAULT_GRID_SIZE_X,
        art.DEFAULT_GRID_SIZE_Y,
    )
    sigma_min = art.block_nonnormal_singular_value_grid(
        eigenvalues,
        pairs,
        singles,
        art.NONNORMAL_COUPLING,
        z_grid,
    )

    figure, axis = plt.subplots()
    contour_set = axis.contour(x_grid, y_grid, sigma_min, levels=levels)
    contour_groups = [list(collection.get_paths()) for collection in contour_set.collections]
    plt.close(figure)

    if len(contour_groups) != len(levels):
        raise RuntimeError("Matplotlib returned an unexpected number of contour groups.")
    return contour_groups


def contour_bounds(
    contour_groups: Sequence[Sequence[MplPath]],
) -> Tuple[float, float, float, float]:
    vertices = [
        path.vertices
        for group in contour_groups
        for path in group
        if len(path.vertices)
    ]
    if not vertices:
        raise RuntimeError("The contour calculation produced no paths.")
    stacked = np.vstack(vertices)
    finite = stacked[np.all(np.isfinite(stacked), axis=1)]
    if not len(finite):
        raise RuntimeError("The contour paths contain no finite coordinates.")
    return (
        float(np.min(finite[:, 0])),
        float(np.max(finite[:, 0])),
        float(np.min(finite[:, 1])),
        float(np.max(finite[:, 1])),
    )


def render_pdf(
    output_path: Path,
    contour_groups: Sequence[Sequence[MplPath]],
    bounds: Tuple[float, float, float, float],
) -> None:
    """Draw the exact nameplate geometry and vector contour paths."""

    x_min, x_max, y_min, y_max = bounds
    data_width = x_max - x_min
    data_height = y_max - y_min
    scale = min(ART_MAX_WIDTH / data_width, ART_MAX_HEIGHT / data_height)

    data_center_x = 0.5 * (x_min + x_max)
    data_center_y = 0.5 * (y_min + y_max)
    plate_center_x = PLATE_X + 0.5 * PLATE_WIDTH
    plate_center_y = PLATE_Y + 0.5 * PLATE_HEIGHT

    canvas = Canvas(str(output_path), pagesize=letter, pageCompression=1)
    canvas.setTitle("Jonathan Lindbloom - pseudospectrum office nameplate")
    canvas.setSubject("Print-ready 18.8 cm by 3 cm office nameplate on US Letter")
    canvas.setAuthor("Jonathan Lindbloom")
    canvas.setCreator("Python, ReportLab, Matplotlib, and NumPy")

    canvas.setFillColorRGB(0.0, 0.0, 0.0)
    canvas.rect(PLATE_X, PLATE_Y, PLATE_WIDTH, PLATE_HEIGHT, stroke=0, fill=1)

    canvas.saveState()
    clip_path = canvas.beginPath()
    clip_path.rect(PLATE_X, PLATE_Y, PLATE_WIDTH, PLATE_HEIGHT)
    canvas.clipPath(clip_path, stroke=0, fill=0)
    canvas.setLineCap(1)
    canvas.setLineJoin(1)

    for paths, color, linewidth in zip(
        contour_groups,
        DARK_GROUND_CONTOUR_COLORS,
        LINEWIDTHS,
    ):
        canvas.setStrokeColor(HexColor(color))
        canvas.setLineWidth(float(linewidth))
        for contour_path in paths:
            reportlab_path = to_reportlab_path(
                canvas,
                contour_path,
                data_center_x,
                data_center_y,
                plate_center_x,
                plate_center_y,
                scale,
            )
            canvas.drawPath(reportlab_path, stroke=1, fill=0)

    canvas.restoreState()
    canvas.showPage()
    canvas.save()


def to_reportlab_path(
    canvas: Canvas,
    source_path: MplPath,
    data_center_x: float,
    data_center_y: float,
    plate_center_x: float,
    plate_center_y: float,
    scale: float,
):
    """Convert a Matplotlib contour path into page coordinates."""

    target = canvas.beginPath()
    vertices = source_path.vertices
    codes = source_path.codes
    if not len(vertices):
        return target

    def transform(vertex: np.ndarray) -> Tuple[float, float]:
        return (
            plate_center_x + (float(vertex[0]) - data_center_x) * scale,
            plate_center_y + (float(vertex[1]) - data_center_y) * scale,
        )

    if codes is None:
        x_value, y_value = transform(vertices[0])
        target.moveTo(x_value, y_value)
        for vertex in vertices[1:]:
            x_value, y_value = transform(vertex)
            target.lineTo(x_value, y_value)
        return target

    for vertex, code in zip(vertices, codes):
        x_value, y_value = transform(vertex)
        if code == MplPath.MOVETO:
            target.moveTo(x_value, y_value)
        elif code == MplPath.LINETO:
            target.lineTo(x_value, y_value)
        elif code == MplPath.CLOSEPOLY:
            target.close()
        else:
            raise RuntimeError(f"Unsupported contour path code: {code}")
    return target


if __name__ == "__main__":
    main()

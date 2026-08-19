# Jonathan Lindbloom inline OSU pseudospectrum

This directory contains a self-contained remake of the name-as-pseudospectrum artwork.

- `make_jonathan_lindbloom_name_pseudospectrum_original.py` is the source generator copied verbatim from the Overleaf art directory. It produces the original stacked target and several related experiments.
- `make_jonathan_lindbloom_inline_osu_pseudospectrum.py` is the focused redesign. It renders **Jonathan Lindbloom** on one baseline, removes the plot axes, colorbar, and eigenvalue markers, uses a discrete Ohio State scarlet/gray contour scale, and places the darkest contours nearest the eigenvalues. It also renders medium- and high-density alternatives with additional small-epsilon contours concentrated near the eigenvalues.

The original script is standalone: its `.npy`, PDF, and PNG files are outputs, not inputs. The focused script preserves only the code needed to construct the text path, sample the eigenvalues, assemble the local nonnormal blocks, evaluate `sigma_min(zI-A)`, and render the final contour plot.

The wide canvas is derived from the numerical x/y limits rather than imposed on the data. Matplotlib uses an equal data aspect, and the generator verifies at render time that one unit on the real axis occupies exactly the same number of pixels as one unit on the imaginary axis.

## Generate

From this directory:

```powershell
python -m pip install -r requirements.txt
python .\make_jonathan_lindbloom_inline_osu_pseudospectrum.py
```

The command writes:

- `jonathan_lindbloom_inline_osu_pseudospectrum.png`
- `jonathan_lindbloom_inline_osu_pseudospectrum_medium_inner_contours.png`
- `jonathan_lindbloom_inline_osu_pseudospectrum_dense_inner_contours.png`
- `jonathan_lindbloom_inline_osu_pseudospectrum_matrix.npy`
- `../../../assets/images/jonathan-lindbloom-pseudospectrum-inline-osu.png` (the web-facing copy)
- `../../../assets/images/jonathan-lindbloom-pseudospectrum-inline-osu-medium-inner-contours.png` (the medium-density web-facing alternative)
- `../../../assets/images/jonathan-lindbloom-pseudospectrum-inline-osu-dense-inner-contours.png` (the denser web-facing alternative)

Use `--skip-site-copy` when the web-facing copy is not needed. `--grid-x` and `--grid-y` are available for faster draft renders; the defaults are the intended final quality.

The contour colors use Ohio State's published primary and shade values from the [Buckeye UX primary-color reference](https://bux.osu.edu/color/primary-colors/).

## Print-ready office nameplate

Run:

```powershell
python .\make_jonathan_lindbloom_nameplate_pdf.py
```

This creates `../../../output/pdf/jonathan-lindbloom-pseudospectrum-nameplate-letter.pdf`. The PDF is one US Letter page with a solid-black 18.8 cm by 3 cm plate in the same position as the supplied Plain TeX template. It uses the 12-contour medium-density artwork as vectors, reverses the central contours to white and light gray for contrast, and retains scarlet accents toward the outside.

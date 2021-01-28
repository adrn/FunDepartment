# Standard library

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# This project
from .config import plot_path
from .utils import AA
from .data_helpers import apply_masks

FRAME_COLOR = 'tab:purple'
VISIT_COLOR = 'k'
SPEC_STYLE = dict(marker='', ls='-', lw=1, drawstyle='steps-mid')


def plot_spectrum_masked(spectrum):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    wvln = spectrum.wavelength.to_value(AA)
    flux = spectrum.flux.value

    ax.plot(wvln[~spectrum.mask],
            flux[~spectrum.mask],
            **SPEC_STYLE)

    ax.plot(wvln[spectrum.mask],
            flux[spectrum.mask],
            marker='o', mew=0, ms=2., ls='none',
            color='tab:red', alpha=0.75, zorder=-10)

    ax.set_xlim(wvln.min(), wvln.max())

    fmin, fmax = (flux[~spectrum.mask].min(), flux[~spectrum.mask].max())
    ptp = fmax - fmin
    ax.set_ylim(fmin - 0.2*ptp, fmax + 0.2*ptp)

    ax.set_xlabel(f'wavelength [{AA:latex_inline}]')
    ax.set_ylabel('flux')

    return fig


def plot_visit_frames(visit, masked=True, **kwargs):
    spectra = visit.get_frame_spectra(**kwargs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10),
                           constrained_layout=True)

    visit_spec = visit.get_spectrum()
    if masked:
        visit_spec = apply_masks(visit_spec)
    ax.plot(visit_spec.wavelength,
            visit_spec.flux / np.nanmedian(visit_spec.flux),
            color=VISIT_COLOR, **SPEC_STYLE)

    for i, (frame, s) in enumerate(spectra.items()):
        if masked:
            s = apply_masks(s)
        ax.plot(s.wavelength.value,
                s.flux / np.nanmedian(s.flux) + i + 1,
                color=FRAME_COLOR, **SPEC_STYLE)
        ax.text(s.wavelength.value.min(), 2+i+0.1, str(frame))

    ax.yaxis.set_visible(False)
    ax.set_xlabel(f'wavelength [{AA:latex_inline}]')
    ax.set_title(f"{visit['FILE'].strip()}")

    ax.set_xlim(visit_spec.wavelength.value.min(),
                visit_spec.wavelength.value.max())

    filename = (plot_path /
                f"{visit['APOGEE_ID']}" /
                f"{visit['FILE'][:-5].strip()}-frames-visit.png")
    return fig, filename


def plot_normalized_ref_spectrum(visit, frame_name,
                                 frame_spectrum,
                                 ref_spectrum,
                                 normed_ref_spectrum):

    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             sharex=True,
                             constrained_layout=True)

    for ax in axes:
        ax.plot(frame_spectrum.wavelength.to_value(AA),
                frame_spectrum.flux.value,
                color=FRAME_COLOR, label='frame spectrum',
                **SPEC_STYLE)
        ax.set_ylabel('flux')

    ax = axes[0]
    ax.plot(ref_spectrum.wavelength.to_value(AA),
            ref_spectrum.flux.value,
            color=VISIT_COLOR, label='raw visit spectrum',
            **SPEC_STYLE)
    ax.legend(loc='lower left')
    ax.set_title(f"{visit['FILE'].strip()}, frame={frame_name}")

    ax = axes[1]
    ax.plot(normed_ref_spectrum.wavelength.to_value(AA),
            normed_ref_spectrum.flux.value,
            color=VISIT_COLOR, label='normalized visit spectrum',
            **SPEC_STYLE)
    ax.legend(loc='lower left')
    ax.set_xlabel(f'wavelength [{AA:latex_inline}]')

    filename = (plot_path /
                f"{visit['APOGEE_ID']}" /
                f"{visit['FILE'][:-5].strip()}-{frame_name}.png")

    return fig, filename

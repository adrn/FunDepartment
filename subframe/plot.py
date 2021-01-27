# Standard library

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# This project
from .utils import AA

FRAME_COLOR = 'tab:purple'
VISIT_COLOR = 'k'
SPEC_STYLE = dict(marker='', ls='-', lw=1, drawstyle='steps-mid')


def plot_visit_frames(visit):
    spectra = visit.load_frame_spectra()

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(visit.spectrum.wavelength,
            visit.spectrum.flux / np.nanmedian(visit.spectrum.flux),
            color=VISIT_COLOR, **SPEC_STYLE)

    for i, s in enumerate(spectra.values()):
        ax.plot(s.wavelength,
                s.flux / np.nanmedian(s.flux) + i + 1,
                color=FRAME_COLOR, **SPEC_STYLE)

    return fig


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
    ax.set_title(f"{visit['APOGEE_ID']}, frame={frame_name}")

    ax = axes[1]
    ax.plot(normed_ref_spectrum.wavelength.to_value(AA),
            normed_ref_spectrum.flux.value,
            color=VISIT_COLOR, label='normalized visit spectrum',
            **SPEC_STYLE)
    ax.legend(loc='lower left')
    ax.set_xlabel(f'wavelength [{AA:latex_inline}]')

    return fig


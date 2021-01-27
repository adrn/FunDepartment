# Standard library

# Third-party
from astropy.constants import c as speed_of_light
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from specutils import Spectrum1D

# This project
from .utils import AA


def doppler_factor(dv):
    return np.sqrt((speed_of_light + dv) / (speed_of_light - dv))


def shift_and_interpolate(ref_spectrum, dv, target_wavelength):
    """
    positive dv = shifting reference spectrum to red

    Parameters
    ----------

    """
    shifted_wvln = doppler_factor(dv) * ref_spectrum.wavelength.to_value(AA)
    idx = shifted_wvln.argsort()
    flux_interp = InterpolatedUnivariateSpline(
        shifted_wvln[idx],
        ref_spectrum.flux.value[idx],
        k=3, ext=3)
    return flux_interp(target_wavelength.to_value(AA))


def normalize_ref_to_frame(frame_spectrum, ref_spectrum, deg=4):
    ref_flux_on_frame_grid = shift_and_interpolate(
        ref_spectrum, 0.*u.km/u.s, frame_spectrum.wavelength)

    wvln = frame_spectrum.wavelength.to_value(u.angstrom)
    ref_wvln = np.median(wvln)

    # Design matrix
    d_wvln = wvln - ref_wvln
    terms = [ref_flux_on_frame_grid * d_wvln**i
             for i in range(deg+1)]
    M = np.stack(terms).T

    coeffs = np.linalg.solve(M.T @ M,
                             M.T @ frame_spectrum.flux)

    # Design matrix for (larger) apVisit spectrum
    d_wvln = ref_spectrum.wavelength.to_value(u.angstrom) - ref_wvln
    terms = [ref_spectrum.flux.value * d_wvln**i
             for i in range(deg+1)]
    M = np.stack(terms).T
    normed_spectrum = Spectrum1D(
        M @ coeffs,
        spectral_axis=ref_spectrum.wavelength)

    return normed_spectrum


def cross_correlate(frame_spectrum, normed_ref_spectrum,
                    K_half=2, dv=4.*u.km/u.s, clip_mask=None):

    if clip_mask is None:
        clip_mask = np.zeros(len(frame_spectrum.flux), dtype=bool)

    terms = []
    vs = np.arange(-K_half, K_half+1) * dv
    for v in vs:
        shifted_flux = shift_and_interpolate(normed_ref_spectrum,
                                             v,
                                             frame_spectrum.wavelength)
        terms.append(shifted_flux)
    M = np.stack(terms).T

    Cinv = 1 / frame_spectrum.uncertainty.array**2
    Cinv[clip_mask] = 0

    denom = np.sqrt(np.diag((M.T * Cinv) @ M) *
                    ((frame_spectrum.flux.T * Cinv) @ frame_spectrum.flux))
    crosscorr = ((M.T * Cinv) @ frame_spectrum.flux) / denom

    return crosscorr, vs


def estimate_kernel(frame_spectrum, normed_ref_spectrum,
                    K_half=2, dv=4.*u.km/u.s, clip_mask=None):

    if clip_mask is None:
        clip_mask = np.zeros(len(frame_spectrum.flux), dtype=bool)

    terms = []
    vs = np.arange(-K_half, K_half+1) * dv
    for v in vs:
        shifted_flux = shift_and_interpolate(normed_ref_spectrum,
                                             v,
                                             frame_spectrum.wavelength)
        terms.append(shifted_flux)
    M = np.stack(terms).T

    # Cinv = np.ones(len(frame_spectrum.flux))
    Cinv = 1 / frame_spectrum.uncertainty.array**2
    Cinv[clip_mask] = 0

    kernel = np.linalg.solve((M.T * Cinv) @ M,
                             (M.T * Cinv) @ frame_spectrum.flux)
    kernel_cov = np.linalg.inv((M.T * Cinv) @ M)

    return kernel, kernel_cov, vs

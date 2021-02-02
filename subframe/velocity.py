# Standard library

# Third-party
from astropy.constants import c as speed_of_light
from astropy.nddata import StdDevUncertainty
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from specutils import Spectrum1D

# This project
from .utils import WVLNU, wavelength_chip_index
from .log import logger


def doppler_factor(dv):
    return np.sqrt((speed_of_light + dv) / (speed_of_light - dv))


def shift_and_interpolate(template_spectrum, dv, spectrum):
    """
    Shifts template spectrum by ``dv``, then interpolates onto the input
    ``spectrum``'s wavelength grid.

    positive dv = shifting template spectrum to red

    Parameters
    ----------

    """
    template_wvln = template_spectrum.wavelength.to_value(WVLNU)
    shifted_wvln = doppler_factor(dv) * template_wvln

    idx = shifted_wvln.argsort()
    wvln = shifted_wvln[idx]
    flux = template_spectrum.flux.value[idx]

    flux_interp = InterpolatedUnivariateSpline(
        wvln,
        flux,
        k=3, ext=3)

    new_spectrum = Spectrum1D(
        flux_interp(spectrum.wavelength.to_value(WVLNU)) * u.one,
        spectral_axis=spectrum.wavelength)

    return new_spectrum


def polynomial_design_matrix(spectrum, deg=3):
    wvln = 30 * (spectrum.wavelength - 1.6*u.micron).to_value(WVLNU)  # MAGIC
    return np.vander(wvln, N=deg+1)


def polynomial_fit_to_template(spectrum, template_spectrum,
                               dv=0*u.km/u.s, deg=3):
    """
    In current usage, template = aspcapStar

    Note: assuming that template spectrum is much higher S/N, so we
    ignore the inv. variance array for the template.

    TODO: audit for ref/reference -> template

    TODO: suspiciously, this looks *very* sensitive to deg
    """

    if spectrum.mask is not None:  # this is v aggro
        raise RuntimeError("Apply mask before passing it in")

    interp_template_s = shift_and_interpolate(template_spectrum,
                                              dv,
                                              spectrum)

    M = polynomial_design_matrix(spectrum, deg=deg)
    Mf = M * interp_template_s.flux.value[:, None]
    flux_err = spectrum.uncertainty.array
    Cinv = 1 / flux_err ** 2

    chip_ids = wavelength_chip_index(spectrum)
    new_flux = np.zeros(spectrum.shape[0])
    new_flux_err = np.zeros(spectrum.shape[0])
    for _id in np.unique(chip_ids):
        chip_mask = chip_ids == _id

        mf = Mf[chip_mask]
        c = Cinv[chip_mask]
        y = spectrum.flux[chip_mask].value
#         coeffs = np.linalg.solve((mf.T * c) @ mf,
#                                  (mf.T * c) @ y)
        coeffs, *_ = np.linalg.lstsq((mf.T * c) @ mf,
                                     (mf.T * c) @ y,
                                     rcond=1e-15)
        logger.debug("Condition number: {:.1e}".format(
            np.linalg.cond((mf.T * c) @ mf)))

        new_flux[chip_mask] = y / (M[chip_mask] @ coeffs)
        new_flux_err[chip_mask] = flux_err[chip_mask] / (M[chip_mask] @ coeffs)

    new_spectrum = Spectrum1D(
        new_flux * u.one,
        spectral_axis=spectrum.wavelength,
        uncertainty=StdDevUncertainty(new_flux_err))

    return new_spectrum, interp_template_s


def cross_correlate(frame_spectrum, normed_ref_spectrum,
                    K_half=1, dv=8.*u.km/u.s, clip_mask=None):

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
                    K_half=2, dv=8.*u.km/u.s, clip_mask=None):

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


def bag_of_hacks_cross_correlate(normed_frame_spectrum, template_spectrum,
                                 K_half=5, dv=8.*u.km/u.s, v0=0*u.km/u.s):

    vs = np.arange(-K_half, K_half+1) * dv + v0
    terms = []
    for dv in vs:
        shifted_template = shift_and_interpolate(
            template_spectrum, dv, normed_frame_spectrum)
        terms.append(shifted_template.flux.value)

    M = np.stack(terms).T

    Cinv = 1 / normed_frame_spectrum.uncertainty.array ** 2

    frame_flux = normed_frame_spectrum.flux
    denom = np.sqrt(np.diag((M.T * Cinv) @ M) *
                    ((frame_flux.T * Cinv) @ frame_flux))
    cc = ((M.T * Cinv) @ normed_frame_spectrum.flux) / denom

    return cc, vs, M


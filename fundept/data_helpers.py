"""
Utilities for downloading, caching, loading APOGEE data
"""
# Standard library
from copy import copy
import os
from collections import defaultdict

# Third-party
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.stats import sigma_clip
import numpy as np
import requests
from specutils import Spectrum1D

# This project
from .config import dr, reduction, cache_path, sdss_auth
from .log import logger


SAS_URL = "https://data.sdss.org/sas/"


def _authcheck():
    if sdss_auth is None:
        raise RuntimeError("No SDSS authentication information available. "
                           "Create a ~/.sdss.login file with the standard "
                           "SDSS username and password (one per line).")


def get_apStar(visit):
    _authcheck()
    if dr == 'dr17alpha':
        stars = 'stars.0.17.12'
    else:
        stars = 'stars'

    root_path = f"apogeework/apogee/spectro/redux/{dr}/{stars}/"

    if visit['TELESCOPE'] == 'apo25m':
        sorp = 'p'
    elif visit['TELESCOPE'] == 'lco25m':
        sorp = 's'
    else:
        raise NotImplementedError()

    sub_path = (f"{visit['TELESCOPE']}/" +
                f"{visit['FIELD'].strip()}/")
    filename = f"a{sorp}Star-{reduction}-{visit['APOGEE_ID'].strip()}.fits"
    url = os.path.join(SAS_URL, root_path, sub_path, filename)

    local_path = cache_path / visit['APOGEE_ID']
    local_path.mkdir(exist_ok=True, parents=True)

    local_file = local_path / filename
    if not local_file.exists():
        logger.debug(f"downloading {url} ...")
        r = requests.get(url, auth=sdss_auth)

        if not r.ok:
            raise RuntimeError(f"Failed to download file from {url}: {r}")

        with open(local_file, 'wb') as f:
            f.write(r.content)

    hdul = fits.open(local_file)
    return hdul


def get_aspcapStar(visit):
    _authcheck()

    root_path = f"apogeework/apogee/spectro/aspcap/{dr}/l33/"
    sub_path = (f"{visit['TELESCOPE']}/" +
                f"{visit['FIELD'].strip()}/")
    filename = f"aspcapStar-{reduction}-{visit['APOGEE_ID'].strip()}.fits"
    url = os.path.join(SAS_URL, root_path, sub_path, filename)

    local_path = cache_path / visit['APOGEE_ID']
    local_path.mkdir(exist_ok=True, parents=True)

    local_file = local_path / filename
    if not local_file.exists():
        logger.debug(f"downloading {url} ...")
        r = requests.get(url, auth=sdss_auth)

        if not r.ok:
            raise RuntimeError(f"Failed to download file from {url}: {r}")

        with open(local_file, 'wb') as f:
            f.write(r.content)

    hdul = fits.open(local_file)
    return hdul


def get_apVisit(visit):
    _authcheck()
    root_path = f"apogeework/apogee/spectro/redux/{dr}/visit/"

    if visit['TELESCOPE'] == 'apo25m':
        sorp = 'p'
    elif visit['TELESCOPE'] == 'lco25m':
        sorp = 's'
    else:
        raise NotImplementedError()

    sub_path = (f"{visit['TELESCOPE']}/" +
                f"{visit['FIELD'].strip()}/" +
                f"{int(visit['PLATE']):04d}/" +
                f"{int(visit['MJD']):05d}/")
    filename = (f"a{sorp}Visit-{reduction}-{int(visit['PLATE']):04d}-" +
                f"{int(visit['MJD']):05d}-" +
                f"{int(visit['FIBERID']):03d}.fits")
    url = os.path.join(SAS_URL, root_path, sub_path, filename)

    local_path = cache_path / visit['APOGEE_ID']
    local_path.mkdir(exist_ok=True, parents=True)

    local_file = local_path / filename
    if not local_file.exists():
        logger.debug(f"downloading {url} ...")
        r = requests.get(url, auth=sdss_auth)

        if not r.ok:
            raise RuntimeError(f"Failed to download file from {url}: {r}")

        with open(local_file, 'wb') as f:
            f.write(r.content)

    hdul = fits.open(local_file)
    return hdul


def get_apCframes(visit):
    _authcheck()
    root_path = f"apogeework/apogee/spectro/redux/{dr}/visit/"

    if visit['TELESCOPE'] == 'apo25m':
        sorp = 'p'
    elif visit['TELESCOPE'] == 'lco25m':
        sorp = 's'
    else:
        raise NotImplementedError()

    sub_path = (f"{visit['TELESCOPE']}/" +
                f"{visit['FIELD'].strip()}/" +
                f"{int(visit['PLATE']):04d}/" +
                f"{int(visit['MJD']):05d}/")

    visit_hdul = get_apVisit(visit)
    frames = [int(visit_hdul[0].header[k]) for k in visit_hdul[0].header.keys()
              if k.startswith('FRAME')]

    if len(frames) <= 1:
        return None

    hduls = defaultdict(dict)
    for frame in frames:
        for chip in ['a', 'b', 'c']:
            filename = f'a{sorp}Cframe-{chip}-{frame:08d}.fits'

            url = os.path.join(SAS_URL, root_path, sub_path, filename)

            local_path = (cache_path /
                          f"{int(visit['PLATE']):04d}" /
                          f"{int(visit['MJD']):05d}/")
            local_path.mkdir(exist_ok=True, parents=True)

            local_file = local_path / filename
            if not local_file.exists():
                logger.debug(f"downloading {url} ...")
                r = requests.get(url, auth=sdss_auth)

                if not r.ok:
                    raise RuntimeError(
                        f"Failed to download file from {url}: {r}")

                with open(local_file, 'wb') as f:
                    f.write(r.content)

            hduls[frame][chip] = fits.open(local_file)

    return hduls


def aggro_percentile_clip(spectrum, poly_deg=5, clip_percentile=99, grow=2):
    # initial pass to get rid of crazy outliers
    init_flux = sigma_clip(spectrum.flux, sigma=5)
    wvln = spectrum.wavelength.value
    ivar = 1 / spectrum.uncertainty.quantity.value ** 2

    if spectrum.mask is not None:
        mask = spectrum.mask
    else:
        mask = np.zeros(len(wvln))
    mask |= init_flux.mask | (~np.isfinite(ivar)) | (ivar < 1e-14)

    coeffs = np.polyfit(
        wvln[~mask],
        init_flux[~mask],
        w=ivar[~mask],
        deg=poly_deg)
    continuum_poly = np.poly1d(coeffs)

    diff = spectrum.flux.value - continuum_poly(wvln)
    clip_mask = diff > np.percentile(diff[~mask], clip_percentile)

    if grow > 0:
        for shift in np.arange(-grow, grow+1):
            clip_mask |= np.roll(clip_mask, shift=shift)

    mask |= clip_mask

    return Spectrum1D(
        flux=spectrum.flux[~mask],
        spectral_axis=spectrum.wavelength[~mask],
        uncertainty=StdDevUncertainty(spectrum.uncertainty[~mask])), mask


def clean_spectrum(spectrum, grow=2, percentile_clip=None):

    if grow is not False and spectrum.mask is not None:
        clip_mask = spectrum.mask.copy()
        for shift in np.arange(-grow, grow+1):
            clip_mask |= np.roll(clip_mask, shift=shift)

        spectrum = copy(spectrum)
        spectrum.mask = clip_mask

    if percentile_clip is not False:
        if percentile_clip in [True, None]:
            kw = {}
        else:
            kw = dict(percentile_clip)

        _, mask = aggro_percentile_clip(spectrum, **kw)
        spectrum = Spectrum1D(
            flux=spectrum.flux,
            spectral_axis=spectrum.wavelength,
            uncertainty=spectrum.uncertainty,
            mask=mask)

    return spectrum


def apply_masks(*spectra, fillnan=False):
    """

    """

    combined_mask = np.bitwise_or.reduce([
        s.mask if s.mask is not None else False for s in spectra],
        axis=0)

    if isinstance(combined_mask, bool):
        combined_mask = np.zeros(spectra[0].shape, dtype=bool)

    masked_spectra = []
    for spectrum in spectra:
        if spectrum.uncertainty is not None:
            err = spectrum.uncertainty.quantity
        else:
            err = None

        flux = spectrum.flux.copy()
        wvln = spectrum.wavelength.copy()

        if fillnan:
            flux[combined_mask] = np.nan * flux.unit
            if err is not None:
                unc = StdDevUncertainty(err)
        else:
            flux = flux[~combined_mask]
            wvln = wvln[~combined_mask]

            if err is not None:
                unc = StdDevUncertainty(err[~combined_mask])

        if err is None:
            unc = None

        spectrum = Spectrum1D(
            flux=flux,
            spectral_axis=wvln,
            uncertainty=unc)

        if len(spectra) == 1:
            return spectrum

        masked_spectra.append(spectrum)

    return masked_spectra

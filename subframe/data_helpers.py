"""
Utilities for downloading, caching, loading APOGEE data
"""
# Standard library
import os
from collections import defaultdict

# Third-party
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
import astropy.units as u
import numpy as np
import requests
from specutils import Spectrum1D

# This project
from .config import cache_path, sdss_auth
from .log import logger
from .utils import combine_spectra


SAS_URL = "https://data.sdss.org/sas/"


def _authcheck():
    if sdss_auth is None:
        raise RuntimeError("No SDSS authentication information available. "
                           "Create a ~/.sdss.login file with the standard "
                           "SDSS username and password (one per line).")


def get_apVisit(visit):
    _authcheck()
    root_path = "apogeework/apogee/spectro/redux/dr16/visit/"

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
    filename = (f"a{sorp}Visit-r12-{int(visit['PLATE']):04d}-" +
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
    root_path = "apogeework/apogee/spectro/redux/dr16/visit/"

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
    for chip in ['a', 'b', 'c']:
        for frame in frames:
            filename = f'a{sorp}Cframe-{chip}-{frame:08d}.fits'

            url = os.path.join(SAS_URL, root_path, sub_path, filename)

            local_path = cache_path / visit['APOGEE_ID']
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

            hduls[chip][frame] = fits.open(local_file)

    return hduls


def get_visit_spectrum(hdul):
    spectra = []
    for i in range(3):  # chips a, b, c
        mask = (hdul[3].data[i] % 15) > 0
        flux_err = hdul[2].data[i] * u.one
        unc = StdDevUncertainty(flux_err)

        s = Spectrum1D(
            flux=hdul[1].data[i][~mask]*u.one,
            spectral_axis=hdul[4].data[i][~mask]*u.angstrom,
            uncertainty=unc[~mask])

        spectra.append(s)

    spectrum = combine_spectra(*spectra)
    return spectrum


def get_frame_spectrum(hdul, apogee_id, mask_flux=True):
    object_idx, = np.where(hdul[11].data['OBJECT'] == apogee_id)[0]

    flux = hdul[1].data[object_idx]
    flux_err = hdul[2].data[object_idx]
    wvln = hdul[4].data[object_idx]
    mask = (hdul[3].data[object_idx] % 15) > 0

    if mask_flux:
        unc = StdDevUncertainty(flux_err[~mask])
        spectrum = Spectrum1D(flux=flux[~mask]*u.one,
                              spectral_axis=wvln[~mask]*u.angstrom,
                              uncertainty=unc)
    else:
        unc = StdDevUncertainty(flux_err)
        spectrum = Spectrum1D(flux=flux*u.one,
                              spectral_axis=wvln*u.angstrom,
                              uncertainty=unc,
                              mask=mask)

    return spectrum

"""
Utilities for downloading, caching, loading APOGEE data
"""
# Standard library

# Third-party
import astropy.coordinates as coord
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
import astropy.units as u
import numpy as np
from specutils import Spectrum1D

# This project
from .data_helpers import (get_apStar, get_apVisit, get_apCframes,
                           clean_spectrum)
from .utils import combine_spectra


class Visit:

    def __init__(self, visit, sigma_clip_flux=True):
        self._visit_row = visit
        self._apid = self._visit_row['APOGEE_ID']

        self._visit_hdulist = None
        self._frames_hdulist = None

        self.sigma_clip_flux = sigma_clip_flux

    def __getitem__(self, key):
        if key in self.colnames:
            return self._visit_row[key]
        else:
            raise KeyError(f"Invalid key '{key}'")

    @property
    def colnames(self):
        return self._visit_row.colnames

    @property
    def hdulist(self):
        if self._visit_hdulist is None:
            self._visit_hdulist = get_apVisit(self._visit_row)
        return self._visit_hdulist

    @property
    def earth_location(self):
        if self['TELESCOPE'] == 'apo25m':
            return coord.EarthLocation.of_site('APO')
        elif self['TELESCOPE'] == 'lco25m':
            return coord.EarthLocation.of_site('LCO')
        else:
            raise NotImplementedError()

    @property
    def skycoord(self):
        return coord.SkyCoord(self['RA'], self['DEC'], unit='deg')

    @property
    def frame_hdulists(self):
        if self._frames_hdulist is None:
            self._frames_hdulist = get_apCframes(self._visit_row)
        return self._frames_hdulist

    def get_spectrum(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Passed to `clean_spectrum()`, like `percentile_clip=True` or
            `grow=3`.
        """

        hdul = self.hdulist

        spectra = []
        for i in range(3):  # chips a, b, c
            flux = hdul[1].data[i]
            flux_err = hdul[2].data[i] * u.one
            mask = (hdul[3].data[i] != 0) | np.isnan(flux)
            wvln = hdul[4].data[i]

            s = Spectrum1D(
                flux=flux * u.one,
                spectral_axis=wvln * u.angstrom,
                uncertainty=StdDevUncertainty(flux_err),
                mask=mask)
            spectra.append(s)

        spectrum = combine_spectra(*spectra, sort=True)

        return clean_spectrum(spectrum, **kwargs)

    def get_apStar_spectrum(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Passed to `clean_spectrum()`, like `percentile_clip=True` or
            `grow=3`.
        """
        hdul = get_apStar(self)

        i = 0
        flux = hdul[1].data[i]
        flux_err = hdul[2].data[i] * u.one
        mask = (hdul[3].data[i] != 0) | np.isnan(flux)
        wvln = 10 ** (hdul[0].header['CRVAL1'] +
                      np.arange(hdul[1].header['NAXIS1']) *
                      hdul[0].header['CDELT1'])

        s = Spectrum1D(
            flux=flux * u.one,
            spectral_axis=wvln * u.angstrom,
            uncertainty=StdDevUncertainty(flux_err),
            mask=mask)

        return clean_spectrum(s, **kwargs)

    def get_frame_spectra(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            Passed to `clean_spectrum()`, like `percentile_clip=True` or
            `grow=3`.
        """

        spectra = {}
        for frame in self.frame_hdulists:
            chip_spectra = []
            for chip in self.frame_hdulists[frame]:
                hdul = self.frame_hdulists[frame][chip]

                object_idx, = np.where(
                    hdul[11].data['OBJECT'] == self['APOGEE_ID'])[0]

                flux = hdul[1].data[object_idx]
                flux_err = hdul[2].data[object_idx]
                wvln = hdul[4].data[object_idx]
                mask = ((hdul[3].data[object_idx] != 0) |
                        (flux <= 0) |
                        np.isnan(flux))

                s = Spectrum1D(
                    flux=flux * u.one,
                    spectral_axis=wvln * u.angstrom,
                    uncertainty=StdDevUncertainty(flux_err),
                    mask=mask)
                chip_spectra.append(s)

            spectrum = combine_spectra(*chip_spectra, sort=True)
            spectra[frame] = clean_spectrum(spectrum, **kwargs)

        return spectra

    @property
    def frame_times(self):
        times = {}
        for frame, hduls in self.frame_hdulists.items():
            hdul = hduls['a']  # any chip - doesn't matter
            times[frame] = Time(hdul[0].header['DATE-OBS'], scale='tai')

        return times

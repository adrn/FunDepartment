"""
Utilities for downloading, caching, loading APOGEE data
"""
# Standard library
from collections import defaultdict

# Third-party
from astropy.time import Time

# This project
from .data_helpers import (get_apVisit, get_apCframes,
                           get_visit_spectrum, get_frame_spectrum)
from .utils import combine_spectra


class Visit:

    def __init__(self, visit):
        self._visit_row = visit
        self._apid = self._visit_row['APOGEE_ID']

        self._visit_hdulist = None
        self._frames_hdulist = None

    @property
    def hdulist(self):
        if self._visit_hdulist is None:
            self._visit_hdulist = get_apVisit(self._visit_row)
        return self._visit_hdulist

    @property
    def spectrum(self):
        return get_visit_spectrum(self.hdulist)

    @property
    def frame_hdulists(self):
        if self._frames_hdulist is None:
            self._frames_hdulist = get_apCframes(self._visit_row)
        return self._frames_hdulist

    def load_frame_spectra(self, stitch=True, mask_flux=True):
        if stitch is False:
            raise NotImplementedError()

        flipped = defaultdict(dict)
        for key, val in self.frame_hdulists.items():
            for subkey, subval in val.items():
                flipped[subkey][key] = subval

        spectra = {}
        for frame in flipped:
            chip_spectra = []
            for chip in flipped[frame]:
                spec = get_frame_spectrum(flipped[frame][chip],
                                          self._apid,
                                          mask_flux=mask_flux)
                chip_spectra.append(spec)

            spectra[frame] = combine_spectra(*chip_spectra)

        return spectra

    @property
    def frame_times(self):
        hduls = self.frame_hdulists['a']  # any chip - doesn't matter

        times = {}
        for frame, hdul in hduls.items():
            times[frame] = Time(hdul[0].header['DATE-OBS'], scale='tai')

        return times

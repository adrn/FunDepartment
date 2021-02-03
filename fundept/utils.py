import numpy as np
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from specutils import Spectrum1D

AA = u.angstrom
WVLNU = u.micron


def combine_spectra(*spectra, sort=True):

    data = {
        'spectral_axis': [],
        'flux': [],
        'mask': [],
        'uncertainty': []
    }
    units = {}

    for s in spectra:
        for k in data:
            val = getattr(s, k)

            if hasattr(val, 'quantity'):
                val = val.quantity

            if hasattr(val, 'unit'):
                if k not in units:
                    units[k] = val.unit
                val = val.to_value(units[k])

            if val is not None:
                data[k].append(val)

    for k in data:
        if len(data[k]) < 1:
            data[k] = None
            continue

        data[k] = np.concatenate(data[k])
        if k in units:
            data[k] = data[k] * units[k]

        if k == 'uncertainty':
            data[k] = StdDevUncertainty(data[k])

    if sort:
        idx = np.argsort(data['spectral_axis'])
        for k in data:
            if data[k] is None:
                continue
            data[k] = data[k][idx]

    return Spectrum1D(**data)


def parabola_optimum(x, y, fit_half_size=1):
    ctr_i = np.argmax(y)
    if ctr_i in [0, len(y)-1]:
        # FAILURE
        return np.nan, None, None

    ps = np.polyfit(x[ctr_i-fit_half_size:ctr_i+fit_half_size+1],
                    y[ctr_i-fit_half_size:ctr_i+fit_half_size+1],
                    deg=2)
    poly = np.poly1d(ps)

    x0 = -ps[1] / (2 * ps[0])

    if ps[0] > 0:
        # FAILURE
        return np.nan, poly, ctr_i

    return x0, poly, ctr_i


def wavelength_chip_index(spectrum):
    """
    Note: This isn't always *exactly* correct because of rest-frame bullshit
    """
    chip_gaps = [0, 1.583, 1.69, 2] * u.micron

    ids = np.zeros(spectrum.shape[0], dtype=int)
    for i, (l, r) in enumerate(zip(chip_gaps[:-1], chip_gaps[1:])):
        ids[(spectrum.wavelength > l) &
            (spectrum.wavelength <= r)] = i

    return ids

import numpy as np
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from specutils import Spectrum1D

AA = u.angstrom


def combine_spectra(*spectra):

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

    return Spectrum1D(**data)


def parabola_optimum(x, y):
    ctr_i = np.argmax(y)
    if ctr_i in [0, len(y)-1]:
        # FAILURE
        return np.nan, None

    ps = np.polyfit(x[ctr_i-1:ctr_i+2],
                    y[ctr_i-1:ctr_i+2],
                    deg=2)
    poly = np.poly1d(ps)

    x0 = -ps[1] / (2 * ps[0])

    if ps[0] > 0:
        # FAILURE
        return np.nan, poly

    return x0, poly

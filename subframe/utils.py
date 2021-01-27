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
    ps = np.polyfit(x, y, deg=2)
    x0 = -ps[1] / (2*ps[0])
    return x0, np.poly1d(ps)

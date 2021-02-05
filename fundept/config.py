import os
import pathlib
import socket


allstar_path = pathlib.Path('~/data/APOGEE_DR16/allStar-r12-l33.fits')
# allstar_path = pathlib.Path(
#     '~/data/APOGEE_DR17/allStarLite-dr17-l33alpha.fits')
allstar_path = allstar_path.expanduser()

allvisit_path = pathlib.Path('~/data/APOGEE_DR16/allVisit-r12-l33.fits')
# allvisit_path = pathlib.Path(
#     '~/data/APOGEE_DR17/allVisitLite-dr17-l33alpha.fits')
allvisit_path = allvisit_path.expanduser()

dr = 'dr16'
reduction = 'r12'
# dr = 'dr17'
# reduction = 'dr17'

ROOT_CACHE_PATH = pathlib.Path(
    os.environ.get("APOGEE_CACHE_PATH",
                   os.path.join("~", ".apogee"))).expanduser()

cache_path = ROOT_CACHE_PATH / dr
cache_path.mkdir(parents=True, exist_ok=True)

if socket.gethostname().startswith('worker'):  # rusty
    ROOT_PLOT_PATH = pathlib.Path('~/public_www/plots/apogee-subframe')
    plot_path = ROOT_PLOT_PATH.expanduser() / dr
    plot_path.mkdir(parents=True, exist_ok=True)

else:  # force it to bork
    plot_path = ROOT_CACHE_PATH / 'plots'

# Load authentication for SDSS
sdss_auth_file = pathlib.Path('~/.sdss.login').expanduser()
if sdss_auth_file.exists():
    with open(sdss_auth_file, 'r') as f:
        sdss_auth = f.readlines()
    sdss_auth = tuple([s.strip() for s in sdss_auth if len(s.strip()) > 0])
else:
    sdss_auth = None

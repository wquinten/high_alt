import pandas as pd
from herbie import Herbie
import xarray as xr
import numpy as np
import metpy
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import warnings
import traceback

# Herbie Configuration
MODEL_NAME = 'gfs'
PRODUCT_NAME = 'pgrb2.1p00'
AVAILABLE_LEVELS = np.array([250,200,150, 100, 70, 50, 30, 20, 10])
DATA_LAT_MIN = -85
DATA_LAT_MAX = 85

def pressure_to_flight_level(pressure_hpa):
    if pressure_hpa <= 0:
        return 'N/A'
    
    try:
        p0 = 1013.25
        altitude_ft = (1 - (pressure_hpa/p0))**(1/5.255) * 145366.45
        fl = round(altitude_ft/100)
        return f'FL{fl:03d}' 
    except Exception:
        return 'N/A'
    
def vertical_wind_shear(u_levels, v_levels, hgt_levels):
    if len(u_levels.isobaricInhPa) != 2 or len(v_levels.isobaricInhPa) != 2 or len(hgt_levels.isobaricInhPa) != 2:
        raise ValueError('Input DataArrays for vws func must have exactly two pressure levels')
    
    u1,v1,z1 = u_levels.isel(isobaricInhPa=0),v_levels.isel(isobaricInhPa=0),hgt_levels.isel(isobaricInhPa=0) # check to see if this 0 and 1 are accurate
    u2,v2,z2 = u_levels.isel(isobaricInhPa=1),v_levels.isel(isobaricInhPa=1),hgt_levels.isel(isobaricInhPa=1)

    du, dv, dz = u2 - u1, v2 - v1, z2 - z1

    # add some checking to make sure dz is not zero and is not negative?

    vws = np.sqrt((du/dz)**2 + (dv/dz)**2)

    return vws

def get_adjacent_levels(target_lev, available_levs):
    if target_lev not in available_levs: raise ValueError(f'Target levels {target_lev} hPa not in available GFS levels.')
    if not np.all(np.diff(available_levs) < 0): raise ValueError('Available levels must be sorted descendingly.')

    idx = list(available_levs).index(target_lev)
    higher_lev = available_levs[idx + 1] if idx + 1 < len(available_levs) else None
    lower_lev = available_levs[idx - 1] if idx - 1 >= 0 else None
     
    
    return higher_lev, lower_lev

def get_and_calculate_EI(date_str, model_cycle_int, target_lev, fxx, lat_min, lat_max):
    model_run_dt = pd.to_datetime(f'{date_str} {model_cycle_int:02d}:00')
    print(f'Fetching GFS: {model_run_dt}, F{fxx:03d}, Product: {PRODUCT_NAME}')
    
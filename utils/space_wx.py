import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta, datetime, timezone
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import re
import io


def GOES_plot():

    GOES_results = []

    for tool in ['primary','secondary']:
        url = f'https://services.swpc.noaa.gov/json/goes/{tool}/integral-protons-plot-3-day.json'
        resp = requests.get(url)
        resp.raise_for_status()

        df = pd.DataFrame(resp.json())

        GOES_results.append(df)

    GOES_data = pd.concat(GOES_results, ignore_index=True)
    GOES_data["time_tag"] = pd.to_datetime(GOES_data["time_tag"], utc=True) 

    GOES18 = GOES_data[GOES_data['satellite'] == 18]
    
    fig, ax = plt.subplots(figsize=(12,6))
    
   
    for energy_level in ['>=10 MeV', '>=50 MeV', '>=100 MeV', '>=500 MeV']:
        subset = GOES18[GOES18['energy'] == energy_level]
        ax.plot(subset['time_tag'], subset['flux'], label=energy_level)
    
    #for sat_id, group in GOES_data[GOES_data['energy'] == '>=10 MeV'].groupby('satellite'):
    #    ax.plot(group['time_tag'], group['flux'], label=f'GOES--{sat_id} (>=10 MeV)')

    ax.set_yscale('log')
    ax.set_ylim(1e-2,1e4)
    ax.set_xlim(GOES_data['time_tag'].min(),GOES_data['time_tag'].max() + timedelta(hours=6))
    ax.set_title('GOES Proton Flux Data')
    ax.set_xlabel('UTC time')
    ax.set_ylabel('Flux (pfu)')
    ax.minorticks_off()
    ax.axhline(10, color='red', linestyle='--')
    ax.grid(which='both', linestyle=':')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()

    return fig


def NAIRAS_plot():

    base_url = 'https://iswa.gsfc.nasa.gov/iswa_data_tree/model/radiation_and_plasma_effects/NAIRAS/NAIRAS-Dose-Data/EffectiveDose'

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    max_lookback_hrs = 3

    for i in range(max_lookback_hrs + 1):
        candidate = now -timedelta(hours=i)
        datestr = candidate.strftime('%Y%m%d_%H0000')

        url = f'{base_url}/{now.year}/{now.month:02d}/{datestr}_EffectiveDose20km.json'

        try: 
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                model_time = candidate
                break
            
        except requests.RequestException:
            pass

    df = pd.DataFrame(data)

    df['effective_dose']  = df['effective_dose']/10

    lats = df['lat']
    lons = df['lon']
    dose = df['effective_dose']

    max_dosage = max(df['effective_dose'])


    lons = np.where(lons > 180, lons - 360, lons)

    grid_lon = np.linspace(min(lons), max(lons), 100)
    grid_lat = np.linspace(min(lats), max(lats), 100)

    X, Y = np.meshgrid(grid_lon, grid_lat)

    Z = griddata((lons, lats), dose, (X,Y), method='linear')

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())

    c = ax.contour(X, Y, Z, levels=5, colors='blue', linewidths=1, transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    plt.clabel(c, inline=True, fontsize=10, fmt='%1.1f')
    ax.annotate(f'Max Effective Dose: {max_dosage:.2f}',
                xy=(0.01,0.95), xycoords='axes fraction', color='blue')
    ax.annotate(f'Valid: {model_time.strftime('%-m/%-d/%y at %H:%MZ')}',
                xy=(0.99,0.95), xycoords='axes fraction', ha='right', color='blue')
    ax.set_title('Radiation Effective Dose at FL660')


    return fig, max_dosage



def get_Kp():

    r = requests.get('https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json')
    r.raise_for_status()

    df = pd.DataFrame(r.json()[1:], columns=r.json()[0])
    
    df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)
    df['Kp'] = df['Kp'].astype(float)

    fig, ax = plt.subplots(figsize=(12,6))

    ax.bar(df['time_tag'].iloc[-18:], df['Kp'].iloc[-18:], width=.1, edgecolor='black')
    ax.set_title('Estimated Planetary K Index (3-hour data)')
    ax.set_ylabel('Kp Index')
    ax.set_ylim(0,9)
    ax.set_xlabel('UTC')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0,6,12,18]))
    ax.grid(axis='y', which='major', ls=':', lw=.6)
    ax.axhline(5, color='red', linestyle='--')

    current_Kp = df['Kp'].iloc[-1]

    return  fig, current_Kp

def get_wind_magnet():

    solar_wind_url = 'https://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json'
    magnet_field_url = 'https://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json'

    r_wind = requests.get(solar_wind_url)
    r_wind.raise_for_status()

    r_magnet = requests.get(magnet_field_url)
    r_magnet.raise_for_status()

    solar_wind = pd.DataFrame(r_wind.json()[1:], columns=r_wind.json()[0])
    magnet = pd.DataFrame(r_magnet.json()[1:], columns=r_magnet.json()[0])

    solar_wind['time_tag'] = pd.to_datetime(solar_wind['time_tag'], utc=True)
    magnet['time_tag'] = pd.to_datetime(magnet['time_tag'], utc=True)

    solar_wind['speed'] = solar_wind['speed'].astype(float)
    magnet['bz_gsm'] = magnet['bz_gsm'].astype(float)

    solar_fig, solar_ax = plt.subplots(figsize=(12,6))
    solar_ax.plot(solar_wind['time_tag'], solar_wind['speed'])

    magnet_fig, magnet_ax = plt.subplots(figsize=(12,6))
    magnet_ax.plot(magnet['time_tag'], magnet['bz_gsm'])

    solar_recent = round(solar_wind['speed'].iloc[-120:].mean())
    magnet_recent = round(magnet['bz_gsm'].iloc[-120:].mean(),3)

    return solar_fig, magnet_fig, solar_recent, magnet_recent

def get_neutrons():

    stations = ['LMKS','OULU','THUL','MOSC']

    offline_threshold = 15 # minutes

    url = (
        "http://nest.nmdb.eu/draw_graph.php?formchk=1"
        "&stations[]=LMKS&stations[]=OULU&stations[]=INVK&stations[]=THUL"
        "&output=ascii"
        "&tabchoice=revori"
        "&dtype=corr_for_efficiency"
        "&tresolution=0"
        "&yunits=1"
        "&date_choice=last"
        "&last_days=1"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    raw_lines = resp.text.splitlines()

    data_lines = [ln for ln in raw_lines if re.match(r"^\d{4}-\d{2}-\d{2}", ln)]
    data = '\n'.join(data_lines)

    cols = ['time'] + stations

    df = pd.read_csv(
        io.StringIO(data),
        sep=';',
        header=None,
        names=cols,
        na_values=['','null','None'],
        skip_blank_lines=True
    )
    df['time'] = pd.to_datetime(df['time'], utc=True)

    for s in stations:
        df[s] = pd.to_numeric(df[s].astype(str).str.strip(), errors='coerce')
    
    df = df.set_index('time')

    if df.empty:
        return {s: {'offline': True, 'zscore': None} for s in stations}
    
    now = pd.Timestamp.utcnow()
    flags = {}

    last_valid = {s: df[s].last_valid_index() for s in stations}

    last_values = {s: df[s].loc[last_valid[s]] if last_valid[s] else None for s in stations}

    valid_items = {s: val for s, val in last_values.items() if val is not None}
    if len(valid_items) >= 2:
        values = pd.Series(valid_items)
        median = values.median()
        std = values.std()
    else:
        median = None
        std = None

    for s in stations:
        ts = last_valid[s]
        offline = True

        if ts:
            offline = (now - ts) > pd.Timedelta(minutes=offline_threshold)
        val = last_values.get(s)
        if val is None:
            z = None
        elif std and std >  0:
            z = float((val - median) / std)
        else:
            z = None
        flags[s] = {'offline': offline, 'zscore': z}

    return flags
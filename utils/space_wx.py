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

# Main URLs
NOAA_SWPC_BASE_URL = 'https://services.swpc.noaa.gov'
NAIRAS_BASE_URL = 'https://iswa.gsfc.nasa.gov/iswa_data_tree/model/radiation_and_plasma_effects/NAIRAS/NAIRAS-Dose-Data/EffectiveDose'
NMDB_NEUTRON_URL = (
    "http://nest.nmdb.eu/draw_graph.php?formchk=1"
    "&stations[]=LMKS&stations[]=OULU&stations[]=MOSC&stations[]=THUL"
    "&output=ascii"
    "&tabchoice=revori"
    "&dtype=corr_for_efficiency"
    "&tresolution=0"
    "&yunits=1"
    "&date_choice=last"
    "&last_days=1"
)


def GOES_plot():

    GOES_results = []

    for tool in ['primary','secondary']:
        url = f'{NOAA_SWPC_BASE_URL}/json/goes/{tool}/integral-protons-plot-3-day.json'
        resp = requests.get(url)
        resp.raise_for_status()

        df = pd.DataFrame(resp.json())

        GOES_results.append(df)

    GOES_data = pd.concat(GOES_results, ignore_index=True)
    GOES_data["time_tag"] = pd.to_datetime(GOES_data["time_tag"], utc=True) 

    # Also fetched GOES 19 data, filtering down
    GOES18 = GOES_data[GOES_data['satellite'] == 18]
    if GOES18.empty:
        # put fall back method to get GOES19 data in the future
        pass
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    # Plot the various energy levels
    for energy_level in ['>=10 MeV', '>=50 MeV', '>=100 MeV', '>=500 MeV']:
        subset = GOES18[GOES18['energy'] == energy_level]
        ax.plot(subset['time_tag'], subset['flux'], label=energy_level)

    ax.set_yscale('log')
    ax.set_ylim(1e-2,1e4)
    ax.set_xlim(GOES_data['time_tag'].min(),GOES_data['time_tag'].max() + timedelta(hours=6))
    
    ax.set_title('GOES Proton Flux Data')
    ax.set_xlabel('UTC Time')
    ax.set_ylabel('Flux (pfu)')
    ax.minorticks_off()
    ax.axhline(10, color='red', linestyle='--') # shows the alert threshold for > 10 MeV protons
    ax.grid(which='both', linestyle=':')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()

    return fig


def NAIRAS_plot():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    max_lookback_hrs = 3 

    for i in range(max_lookback_hrs + 1):
        candidate = now - timedelta(hours=i)
        datestr = candidate.strftime('%Y%m%d_%H0000')

        url = f'{NAIRAS_BASE_URL}/{candidate.year}/{candidate.month:02d}/{datestr}_EffectiveDose20km.json'
        print(url)
        try: 
            r = requests.get(url, timeout=30) # look into why I chose that timeout
            if r.status_code == 200:
                data = r.json()
                model_time = candidate
                break
            
        except requests.RequestException:
            fig, ax = plt.subplots()
            return fig, 0

    df = pd.DataFrame(data)

    df['effective_dose']  = df['effective_dose']/10 # convert uSv/hr to mrem/hr

    lats = df['lat'].to_numpy()
    lons = df['lon'].to_numpy()
    dose = df['effective_dose'].to_numpy()

    max_dosage = np.nanmax(dose)

    # normalize longitudes to -180 to 180 for plotting the globe
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
                xy=(0.01,0.95), xycoords='axes fraction', color='blue',
                bbox=dict(boxstyle='round,pad=.3', fc='white', ec='blue', lw=1, alpha=.9))
    ax.annotate(f'Valid: {model_time.strftime('%-m/%-d/%y at %H:%MZ')}',
                xy=(0.99,0.95), xycoords='axes fraction', ha='right', color='blue',
                bbox=dict(boxstyle='round,pad=.3', fc='white', ec='blue', lw=1, alpha=.9))
    ax.set_title('Radiation Effective Dose at FL660')
    ax.set_global()


    return fig, max_dosage



def get_Kp():
    
    url = f'{NOAA_SWPC_BASE_URL}/products/noaa-planetary-k-index.json'
    r = requests.get(url)
    r.raise_for_status()

    df = pd.DataFrame(r.json()[1:], columns=r.json()[0])
    
    df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True)
    df['Kp'] = df['Kp'].astype(float)

    fig, ax = plt.subplots(figsize=(12,6))

    ax.bar(df['time_tag'].iloc[-18:], df['Kp'].iloc[-18:], width=.1, edgecolor='black')
    
    ax.set_title('Estimated Planetary K Index (3-hour data)')
    ax.set_ylabel('Kp Index')
    ax.set_ylim(0,9)
    ax.set_xlabel('UTC Time')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0,24,3)))
    ax.grid(axis='y', which='major', ls=':', lw=.6)
    ax.axhline(5, color='red', linestyle='--') # Geomagnetic Storm Threshold
    # Add text saying "Geomagnetic Storm Threshold"

    current_Kp = df['Kp'].iloc[-1]

    return  fig, current_Kp

def get_wind_magnet():

    solar_wind_url = f'{NOAA_SWPC_BASE_URL}/products/solar-wind/plasma-3-day.json'
    magnet_field_url = f'{NOAA_SWPC_BASE_URL}/products/solar-wind/mag-3-day.json'

    r_wind = requests.get(solar_wind_url)
    r_wind.raise_for_status()
    solar_wind = pd.DataFrame(r_wind.json()[1:], columns=r_wind.json()[0])

    r_magnet = requests.get(magnet_field_url)
    r_magnet.raise_for_status()
    magnet = pd.DataFrame(r_magnet.json()[1:], columns=r_magnet.json()[0])

    solar_wind['time_tag'] = pd.to_datetime(solar_wind['time_tag'], utc=True)
    magnet['time_tag'] = pd.to_datetime(magnet['time_tag'], utc=True)

    solar_wind['speed'] = pd.to_numeric(solar_wind['speed'], errors='coerce')
    magnet['bz_gsm'] = pd.to_numeric(magnet['bz_gsm'], errors='coerce')

    solar_fig, solar_ax = plt.subplots(figsize=(12,6))
    solar_ax.plot(solar_wind['time_tag'], solar_wind['speed'])
    solar_ax.set_title('Solar Wind Speed')
    solar_ax.set_ylim([200, 1000])
    solar_ax.set_ylabel('Speed (km/s)')
    solar_ax.set_xlabel('UTC Time')
    solar_ax.axhline(500, color='red', linestyle='--')
    solar_ax.grid(True, linestyle=':', alpha=.7)
    solar_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    magnet_fig, magnet_ax = plt.subplots(figsize=(12,6))
    magnet_ax.plot(magnet['time_tag'], magnet['bz_gsm'])
    magnet_ax.set_title('Interplanetary Magnetic Field (Bz component)')
    magnet_ax.set_ylim([-20, 20])
    magnet_ax.set_ylabel('Bz (nT, GSM)')
    magnet_ax.set_xlabel('UTC Time')
    magnet_ax.axhline(-5, color='red', linestyle='--') 
    magnet_ax.grid(True, linestyle=':', alpha=.7)
    magnet_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    solar_recent = round(solar_wind['speed'].iloc[-120:].mean())
    magnet_recent = round(magnet['bz_gsm'].iloc[-120:].mean(),3)

    return solar_fig, magnet_fig, solar_recent, magnet_recent

def get_neutrons():

    stations = ['LMKS','OULU','THUL','MOSC']

    offline_threshold = 15 # minutes
    min_points = 5 # number of points
    spike_dip = 3 # z-score
    stuck_points = 10 # consecutive points

    try:
        resp = requests.get(NMDB_NEUTRON_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f'Error fetching neutron data: {e}')
        results = {}
        for station in stations:
            results[station] = {
                'offline': True, 'latest_value': None, 'time_since_last_data': None,
                'completeness_last_hour_percent': 0.0, 'completenes_last_day_percent': 0.0,
                'num_anomalous_spikes_hourly': 0, 'num_anomalous_spikes_daily': 0,
                'is_value_stuck': False, 'max_abs_zscore_rate_of_change': None
            }
        return results


    raw_lines = resp.text.splitlines()

    # match lines with data using regex based on the date
    data_lines = []
    for line in raw_lines:
        if re.match(r"^\d{4}-\d{2}-\d{2}", line):
            data_lines.append(line)
    #[ln for ln in raw_lines if re.match(r"^\d{4}-\d{2}-\d{2}", ln)]
    
    # join the lines into singe string for Pandas to read
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

    for station in stations:
        if station in df.columns:
            df[station] = pd.to_numeric(df[station].astype(str).str.strip(), errors='coerce')
    
    df = df.set_index('time')


    station_status_results = {} # to store offline status and z-score
    current_utc = pd.Timestamp.utcnow() # to check how old the last data point is
    one_hour_ago = current_utc - pd.Timedelta(hours=1)

    if df.empty:
        for station in stations:
            station_status_results[stations] = {
                'offline': True, 'latest_value': None, 'time_since_last_data': None,
                'completeness_last_hour_percent': 0.0, 'completeness_last_day_percent': 0.0,
                'max_abs_zscore_hourly': None, 'max_abs_zscore_daily': None,
                'num_anomalous_spikes_hourly': 0, 'num_anomalous_spikes_daily': 0,
                'is_value_stuck': False, 'max_abs_zscore_rate_of_change': None
            }
        return station_status_results
    
    for station in stations:
        res = {
                'offline': True, 'latest_value': None, 'time_since_last_data': None,
                'completeness_last_hour_percent': 0.0, 'completeness_last_day_percent': 0.0,
                'max_abs_zscore_hourly': None, 'max_abs_zscore_daily': None,
                'num_anomalous_spikes_hourly': 0, 'num_anomalous_spikes_daily': 0,
                'is_value_stuck': False, 'max_abs_zscore_rate_of_change': None
        }

        if station not in df.columns:
            station_status_results[station] = res
            continue

        station_readings = df[station].dropna()

        if station_readings.empty:
            station_status_results[station] = res
            continue


        latest_timestamp = station_readings.index[-1]
        res['latest_value'] = station_readings.iloc[-1]
        res['time_since_last_data'] = current_utc - latest_timestamp

        
        if (current_utc - latest_timestamp) <= pd.Timedelta(minutes=offline_threshold):
            res['offline'] = False

        expected_points_hour = 60
        actual_points_hour = station_readings[station_readings.index >= one_hour_ago - pd.Timedelta(minutes=2)].count()
        res['completeness_last_hour_percent'] = round((actual_points_hour / expected_points_hour) * 100, 1)
        print(actual_points_hour)

        expected_points_day = 24 * 60
        actual_points_day = station_readings.count()
        res['completeness_last_day_percent'] = round((actual_points_day / expected_points_day) * 100, 1)

        if res['offline']:
            station_status_results[station] = res
            continue

        if len(station_readings) >= min_points:
            mean_daily = station_readings.mean()
            std_daily = station_readings.std()

            if std_daily is not None and not pd.isna(std_daily) and std_daily > 1e-6:
                zscore_day_series = (station_readings - mean_daily) / std_daily
                if not zscore_day_series.empty:
                    idx_max_abs_daily = zscore_day_series.abs().idxmax()
                    res['max_abs_zscore_daily'] = zscore_day_series.loc[idx_max_abs_daily]
                    res['num_anomalous_spikes_daily'] = np.sum(np.abs(zscore_day_series) > spike_dip)
            
            elif std_daily is not None and not pd.isna(std_daily) and std_daily <= 1e-6:
                if station_readings.nunique() <= 1:
                    res['max_abs_zscore_daily'] = 0.0

        station_readings_hourly = station_readings[station_readings.index >= one_hour_ago]

        if len(station_readings_hourly) >= min_points:
            mean_hourly = station_readings_hourly.mean()
            std_hourly = station_readings_hourly.std()

            if std_hourly is not None and not pd.isna(std_hourly) and std_hourly > 1e-6:
                zscore_hourly_series = (station_readings_hourly - mean_hourly) / std_hourly
                if not zscore_hourly_series.empty:
                    idx_max_abs_hourly = zscore_hourly_series.abs().idxmax()
                    res['max_abs_zscore_hourly'] = zscore_hourly_series.loc[idx_max_abs_hourly]
                    res['num_anomalous_spikes_hourly'] = np.sum(np.abs(zscore_hourly_series) > spike_dip)
            
            elif std_hourly is not None and not pd.isna(std_hourly) and std_hourly <= 1e-6:
                if station_readings_hourly.nunique() <= 1:
                    res['max_abs_zscore_hourly'] = 0.0
        
        if len(station_readings) >= stuck_points:
            recent_values = station_readings.iloc[-stuck_points:]
            if recent_values.nunique() == 1:
                res['is_value_stuck'] = True


        if len(station_readings) >= min_points + 1:
            differences = station_readings.diff().dropna()
            if len(differences) >= min_points:
                mean_diff = differences.mean()
                std_diff = differences.std()
                if std_diff is not None and not pd.isna(std_diff) and std_diff > 1e-6:
                    zscores_diff_series = (differences - mean_diff) / std_diff
                    if not zscores_diff_series.empty:
                        idx_max_abs_diff = zscores_diff_series.abs().idxmax()
                        res['max_abs_zscore_rate_of_change'] = zscores_diff_series.loc[idx_max_abs_diff]
                
                elif std_diff is not None and not pd.isna(std_diff) and std_diff <= 1e-6:
                    if differences.nunique() <= 1:
                        res['max_abs_zscore_rate_of_change'] = 0.0

        station_status_results[station] = res

    return station_status_results 
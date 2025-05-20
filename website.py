import streamlit as st
from utils.space_wx import GOES_plot, NAIRAS_plot, get_Kp, get_wind_magnet, get_neutrons
from datetime import timedelta

st.set_page_config(layout='wide') 

st.title('High Altitude Weather Dashboard')


NARIAS_fig, max_radiation = NAIRAS_plot()
Kp_fig, current_Kp = get_Kp()
wind_fig, mag_fig, wind_recent, mag_recent = get_wind_magnet()
neutrons = get_neutrons()
lmks = neutrons.get("LMKS", {"offline": True, "zscore": None})
oulu = neutrons.get("OULU", {"offline": True, "zscore": None})
mosc = neutrons.get("MOSC", {"offline": True, "zscore": None})
thul = neutrons.get("THUL", {"offline": True, "zscore": None})



st.sidebar.title('Quick Look:')

if max_radiation < 3:
    space_wx_color = '#00ff00'
    text_color = '#000000'
else:
    space_wx_color = '#ff0000'
    text_color = '#000000'


space_wx = st.sidebar.markdown(
    f'''
    <div style="background-color:{space_wx_color}; padding:10px; border-radius:5px;">
        <h4 style="color:{text_color};">Current Space Wx</h4>
        <p style="color:{text_color};">Max Radiation: {round(max_radiation,2)} mrem/hr</p>
        <p style="color:{text_color};">K-value: {str(current_Kp)}</p>
        <p style="color:{text_color};">1hr Avg Solar Wind: {wind_recent} m/s</p>
        <p style="color:{text_color};">1hr Avg Bz: {mag_recent}</p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.sidebar.markdown('---')
st.sidebar.subheader('Neutron Monitor Summary')

num_offline_stations = sum(1 for station in neutrons.values() if station.get('offline', True))
total_stations = len(neutrons)
st.sidebar.metric(label='Online Neutron Monitors', value=f'{total_stations - num_offline_stations}/{total_stations}')

any_stuck = any(s.get('is_value_stuck', False) for s in neutrons.values())
st.sidebar.markdown(f'**Any Stuck Values Detected:** {'Yes' if any_stuck else 'No'}')
    

space_tab, strat_tab = st.tabs(['Space','Stratosphere'])

with space_tab:
    nairas_col, verification_col1, verification_col2 = st.columns(3)

    with nairas_col:
        st.subheader('NAIRAS')
        st.pyplot(NARIAS_fig)
        
        st.subheader('Neutron Monitor Diagnostics')
        
        station_order = ['LMKS', 'OULU', 'THUL', 'MOSC']
        for station in station_order:
            station_data = neutrons.get(station)
            if not station_data:
                st.warning(f'Data for {station} not found.')
                continue
            
            with st.expander(f'Station: {station} {' (Offline)' if station_data.get('offline') else ''}', expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label='Status', value='Offline' if station_data.get('offline') else 'Online')
                    st.metric(label='Last Value', value=f'{station_data.get('latest_value', 'N/A'):.1f}' 
                              if isinstance(station_data.get('latest_value'), (int,float)) else 'N/A')
                    st.metric(label='Last Seen (minutes)', value=f'{(station_data.get('time_since_last_data').total_seconds()/60):.0f}' if not station_data['offline'] else 'Offline')
                    st.metric(label = 'Data Stuck?', value='Yes' if station_data.get('is_value_stuck') else 'No')

                with col2:
                    st.metric(label='Completeness (1hr %)', value=f'{station_data.get('completeness_last_hour_percent', 0.0):.1f}%')
                    st.metric(label='Completeness (1day %)', value=f'{station_data.get('completeness_last_day_percent', 0.0):.1f}%')
                    st.metric(label='Spikes (1hr)', value=str(station_data.get('num_anomalous_spikes_hourly', 0)))
                    st.metric(label='Spikes (1day)', value=str(station_data.get('num_anomalous_spikes_daily', 0)))

                st.markdown('##### Peak Z-Scores (vs. Station\'s Own History)')
                z_col1, z_col2, z_col3 = st.columns(3)
                with z_col1:
                    st.metric(label='Max Z (Hourly)', value=f'{station_data.get('max_abs_zscore_hourly'):.1f}' if not station_data['offline'] else 'Offline', 
                              help='Highest magnitude z-score in the past hour vs. that hour\'s stats for this station.')
                    st.metric(label='Max Z (Daily)', value=f'{station_data.get('max_abs_zscore_daily'):.1f}' if not station_data['offline'] else 'Offline',
                              help='Highest magnitude z-score in the past day vs. that day\'s stats for this station.')
                    st.metric(label='Max Z (Rate of Change)', value=f'{station_data.get('max_abs_zscore_rate_of_change'):.1f}' if not station_data['offline'] else 'Offline',
                              help='Highest magnitude z-score of the change between consecutive points for this station.')
        
        print(neutrons)

    with verification_col1:
        st.subheader('GOES Proton Flux')
        st.pyplot(GOES_plot())
        st.subheader('Kp Plot')
        st.pyplot(Kp_fig)
        
    with verification_col2:
        st.subheader('Solar Wind')
        st.pyplot(wind_fig)
        st.subheader('Bz Values')
        st.pyplot(mag_fig)


with strat_tab:
    st.header('Stratospheric Turbulence Data TBD')


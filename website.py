import streamlit as st
from utils.space_wx import GOES_plot, NAIRAS_plot, get_Kp, get_wind_magnet, get_neutrons

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


space_tab, strat_tab = st.tabs(['Space','Stratosphere'])

with space_tab:
    nairas_col, verification_col1, verification_col2 = st.columns(3)

    with nairas_col:
        st.subheader('NAIRAS')
        st.pyplot(NARIAS_fig)
        st.markdown(
            f'''
            <div style="background-color:{space_wx_color}; padding:10px; border-radius:5px;">
                <h4 style="color:{text_color};">Neutron Monitoring Station Status</h4>
                <p style="color:{text_color};">
                    LMKS: z-score = {round(lmks['z-score_day'],2) if lmks['z-score_day'] is not None else 'NA'}, 
                    offline = {lmks['offline']}
                </p>
                <p style="color:{text_color};">
                    OULU: z-score = {round(oulu['z-score_day'],2) if oulu['z-score_day'] is not None else 'NA'}, 
                    offline = {oulu['offline']}
                </p>
                <p style="color:{text_color};">
                    MOSC: z-score = {round(mosc['z-score_day'],2) if mosc['z-score_day'] is not None else 'NA'}, 
                    offline = {mosc['offline']}
                </p>
                <p style="color:{text_color};">
                    THUL: z-score = {round(thul['z-score_day'],2) if thul['z-score_day'] is not None else 'NA'}, 
                    offline = {thul['offline']}
                </p>
            </div>
            ''',
            unsafe_allow_html=True
        )
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


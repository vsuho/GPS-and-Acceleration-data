import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

path = "Location.csv"
df = pd.read_csv(path)
df_2 = pd.read_csv("Linear Accelerometer.csv")
N = 2  # Poistetaan ensimmäiset 2 riviä
df = df.iloc[N:].reset_index(drop=True)
df_2 = df_2.iloc[N:].reset_index(drop=True)

st.title('Fysiikan loppuprojekti, GPS-ja kiihtyvyysdata')

#Matplotlib kuvaaja
fig, ax = plt.subplots()
ax.plot(df_2['Time (s)'], df_2['Y (m/s^2)'])
ax.set_xlabel('Aika')
ax.set_ylabel('$a_y$')

#Fourier-analyysi ja tehospektri

f = df_2['Y (m/s^2)'] #Signaali
t = df_2['Time (s)'] #Aika
N = len(f) #Havaintojen määrä
dt = np.max(t)/N #Näytteenottoväli (oletetaan vakioksi)

#Fourier-analyysi
fourier = np.fft.fft(f,N) #Fourier-muunnos
psd = fourier*np.conj(fourier)/N #Tehospektri
freq = np.fft.fftfreq(N,dt) #Taajuudet
L = np.arange(1,int(N/2)) #Negatiivisten ja nollataajuuksien rajaus

f_max = freq[L][psd[L] == np.max(psd[L])][0] #Taajuuden arvo, silloin kun tehon arvo saa maksimin. 
T = 1/f_max #Askeleeseen kuluva aika, eli jaksonaika
#askelmäärä = np.max(t)/T
steps = np.max(t)*f_max
print('Dominoiva askeltaajuus on ', f_max)
print('Tätä vastaava jaksonaika (askelaika) ', T)
print('Askelmäärä tällöin {:.2f}'.format(steps))

#st.pyplot(fig)

#Suodatetaan kuvaaja
#Tuodaan filtterifunktiot. Jos scipy puuttuu: pip install scipy

from scipy.signal import butter,filtfilt
def butter_lowpass_filter(data, cutoff, fs, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

#Suodatetaan data
data = df_2['Y (m/s^2)'] #Data
T = df_2['Time (s)'].max() #Koko datan pituus
n = len(df_2['Time (s)']) #datapisteiden lukumäärä
fs = n/T #Näytteenottotaajuus, OLETETAAN VAKIOKSI
nyq = fs/2 
order = 3
cutoff = 2 #cut-off -taajuus, riippuu askeltaajuudesta
filt_signal = butter_lowpass_filter(data, cutoff, fs, nyq, order)

fig, ax = plt.subplots(figsize=(20,6)) #Kuvan koko
#ax.plot(df_2['Time (s)'], df_2['Y (m/s^2)'], label='Alkuperäinen', alpha=0.5)
ax.plot(df_2['Time (s)'], filt_signal, label='Suodatettu') #color='red')

ax.set_xlabel("Aika")
ax.set_ylabel("Suodatettu $a_y$")
ax.legend()

#Lasketaan askelet:
#Tutkitaan, kuinka usein suodatettu signaali ylittää nollatason
jaksot = 0
for i in range(n-1):
    if filt_signal[i]/filt_signal[i+1] < 0: #True jos nollan ylitys, False jos ei ole
        jaksot = jaksot + 1

print('Askelten määrä on ',jaksot/2)

#Tulostetaan lasketut arvot askeleista
st.write(f"Askelmäärä laskettu fourier-analyysin avulla: {steps:.1f}")
st.write(f"Askelmäärä suodatetusta kiihtyvyysdatasta: {jaksot/2:.1f}")
st.write(f"Keskinopeus on: {df['Velocity (m/s)'].mean():.2f} m/s")

# Lasketaan aikavälit
df['Time Diff (s)'] = df['Time (s)'].diff().fillna(0) 
df["Segment Distance (m)"] = df["Velocity (m/s)"] * df["Time Diff (s)"]
Total_distance = df["Segment Distance (m)"].sum()/1000
df['Total Distance (m)'] = df["Segment Distance (m)"].sum()/1000

st.write(f"Kokonaismatka on: {Total_distance:.2f} km")
#Kuljettu matka metreinä jaettuna askelmäärällä = askelpituus
avg_step = Total_distance*1000/240
st.write(f"Askelpituus noin {avg_step:.2f} m")

#Tehospektri
st.title('Tehospektri')
chart_data = pd.DataFrame(np.transpose(np.array([freq[L],psd[L].real])), columns=["freq", "psd"])
st.line_chart(chart_data, x = 'freq', y = 'psd' , y_label = 'Teho',x_label = 'Taajuus [Hz]')

# Näytetään kuvaaja Streamlitissä
st.title('Suodatettu signaali')
st.pyplot(fig)

#Create a map where the center is at start_lat start_long and zoom level is defined
start_lat = df['Latitude (°)'].mean()
start_long = df['Longitude (°)'].mean()
map = folium.Map(location = [start_lat,start_long], zoom_start = 14)

#Piirretään kartta
folium.PolyLine(df[['Latitude (°)', 'Longitude (°)']], color = 'blue', weight = 3.5, opacity = 1).add_to(map)
#Kartan koko
st.title('Karttakuva reitistä')
st_map = st_folium(map, width=900, height=650)

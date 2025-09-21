import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

results = pd.read_csv('csv_cleaned/results.csv')
races = pd.read_csv('csv_cleaned/races.csv')
try:
    weather = pd.read_csv('weather_features_v4.csv')
except:
    weather = pd.DataFrame(columns=['name','round','temperature','precipitation','windspeed'])

df = results.merge(races[['raceId','year','round','name']], on='raceId')
df = df.merge(weather[['name','round','temperature','precipitation','windspeed']], on=['name','round'], how='left')
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
df['points'] = pd.to_numeric(df['points'], errors='coerce')
df[['temperature','precipitation','windspeed']] = df[['temperature','precipitation','windspeed']].fillna(0)
df = df[['grid','points','position','temperature','precipitation','windspeed']].dropna()
df['podium'] = df['position'].apply(lambda x: 1 if x <= 3 else 0)

X = df[['grid','points','temperature','precipitation','windspeed']]
y = df['podium']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

st.title('Prédiction de podium F1')
st.write('Sélectionnez les paramètres pour la simulation')

grid_pos = st.slider('Position sur la grille', 1, 20, 1)
points_scored = st.slider('Points anticipés', 0, 26, 0)
temperature_input = st.slider('Température (°C)', 0.0, 40.0, 20.0)
precip_input = st.slider('Précipitations (mm)', 0.0, 10.0, 0.0)
wind_input = st.slider('Vitesse du vent (km/h)', 0.0, 50.0, 5.0)

if st.button('Prédire'):
    X_new = pd.DataFrame([[grid_pos, points_scored, temperature_input, precip_input, wind_input]], columns=['grid','points','temperature','precipitation','windspeed'])
    prob = model.predict_proba(X_new)[0][1]
    pred = model.predict(X_new)[0]
    st.write('Probabilité de podium :', float(prob))
    if pred == 1:
        st.write('Prévision : Podium')
    else:
        st.write('Prévision : Pas de podium')

st.subheader('Aperçu des données')
st.dataframe(df.head(20))

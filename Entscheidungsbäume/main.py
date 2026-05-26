from sklearn.datasets import load_iris
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

st.title('Präsentation Entscheidungsbaum')

# =============================================================================
# Datensatz importieren
# =============================================================================
df = load_iris(as_frame=True).frame
df_features = df.iloc[:,2:-1]   #Feature Werte für da Training des Entscheidungsbaumes
df_target = df.loc[:,'target']  #Zielwerte für das Training des Entscheidungsbaumes
df_print = df.iloc[:,2:]    #Dataframe zur Visualisierung

dic = dict(enumerate(load_iris().target_names))
df_print['Klasse'] = df_print['target'].map(dic) 

X = df_features
y = df_target







tab1, tab2, tab3 = st.tabs(["Folie 1: Intro", "Folie 2: Iris Datensatz", "Folie 3: Modell"])


with tab1:
    pass
    
with tab2:
    chart = alt.Chart(df_print).mark_circle().encode(x='petal length (cm)',y='petal width (cm)',color='Klasse:N').interactive()
    st.altair_chart(chart)










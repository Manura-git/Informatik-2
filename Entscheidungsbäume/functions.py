from DecisionTree import DecisionTree
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


def Entscheidungsgrenzen(max_depth,min_samples,X,y,target_names=None):
    df_print = X.copy()
    df_print['Klasse'] = y
    
    tree = DecisionTree(max_depth=max_depth, min_samples=min_samples)
    tree.fit(X, y)
    
   #----Meshgrid----#
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    
    # 1. FEINERES RASTER: 0.1 ist zu grob für die feinen Monde. Wir nehmen 0.04
    step = 0.04
    x_range = np.arange(x_min, x_max, step)
    y_range = np.arange(y_min, y_max, step)
    
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Das Grid für die Zeichnung (Linke untere Ecke für Altair)
    mesh_df = pd.DataFrame({
        X.columns[0]: xx.ravel(),
        X.columns[1]: yy.ravel()})
        
    # 2. DER PROFI-TRICK: Ein separates Grid für das Modell (Exakte Mitte des Rechtecks!)
    mesh_df_eval = pd.DataFrame({
        X.columns[0]: xx.ravel() + (step / 2),
        X.columns[1]: yy.ravel() + (step / 2)})
    
    # Wir sagen die Mitte vorher, weisen die Farbe aber dem Zeichen-Grid zu
    mesh_df['Klasse_ID'] = tree.predict(mesh_df_eval)
    
    # Exakte Endpunkte für jedes Rechteck im Grid
    mesh_df['x2'] = mesh_df[X.columns[0]] + step
    mesh_df['y2'] = mesh_df[X.columns[1]] + step
    
    #Color scale 
    unique_classes = list(np.sort(y.unique()))
    color_scale = alt.Scale(domain=unique_classes, scheme='category10')
    
    
    
    
    #Rechtecke
    background = alt.Chart(mesh_df).mark_rect(opacity=0.2).encode(
        x=alt.X(f'{X.columns[0]}:Q', title='0'),
        x2='x2',
        y=alt.Y(f'{X.columns[1]}:Q', title='1'),
        y2='y2',
        color=alt.Color('Klasse_ID:N',scale=color_scale, legend=None),tooltip=alt.value(None))
    
    #dem Dataframe wird eine column mit dem vorhergesagten Werten angehängt (Zum Vergleich im plot)
    df_print['vorhersage'] = tree.predict(X)
    df_print['vorhersage']
    
    #Punkt Chart mit den Objekten
    points = alt.Chart(df_print).mark_circle(size=60, stroke='white', strokeWidth=1).encode(
        x=alt.X('1'),
        y=alt.Y('2'),
        color=alt.Color('Klasse:N',scale=color_scale, title="Echte Klasse"),
        tooltip=['Klasse','vorhersage'])
    
    #Chart aus Punkten und background zusammensetzen
    final_chart = (background + points).properties(
        width=650,
        height=450,
        title=f"ID3 Entscheidungsgebiete (Tiefe: {max_depth})").interactive()
    
    #Chart visualisieren
    st.altair_chart(final_chart, use_container_width=True)
    
    
    
    
    
    
    
    
    
    
    
    
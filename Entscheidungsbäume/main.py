from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
from DecisionTree import _calc_information_gain, _calc_Entropy, DecisionTree, Node
from functions import Entscheidungsgrenzen
import inspect
import random
# =============================================================================
# STREAMLIT VISUALISIERUNG
# =============================================================================

st.set_page_config(layout="wide")

st.title('Präsentation Entscheidungsbaum')

#----IRIS importieren----#

df = load_iris(as_frame=True).frame
df_features = df.iloc[:,2:-1]   #Feature Werte für da Training des Entscheidungsbaumes
df_target = df.loc[:,'target']  #Zielwerte für das Training des Entscheidungsbaumes
df_print = df.iloc[:,2:]    #Dataframe zur Visualisierung

dic = dict(enumerate(load_iris().target_names))
df_print['Klasse'] = df_print['target'].map(dic) 

X = df_features #DataFrame shape=(150, 2)
y = df_target  #Series shape=(150,)





#----Streamlit code----#

tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8, tab9 = st.tabs(["Folie 1: Intro", "Folie 2: Iris Datensatz", "Folie 3: mathematische Grundlage","Folie 4: Visualisierung","Folie5: Code","Folie 6:Trainingsdaten","Folie 7: Training","Folie 8: Baum Visualisierung",'Moons - Beispiel'])


with tab1: 
    st.markdown('''
                ##### Ziel der Präsentation
                - Einen Entscheidungsbaum, auf Basis des ID3 Algorithmus, mit Daten des Iris Datensatz trainieren
                - Parameter wie max_depth und min_samples variieren und die Auswirkungen visualisieren
                
                ''')
    
with tab2:
    st.markdown('##### Iris Datensatz')
    chart = alt.Chart(df_print).mark_circle().encode(x='petal length (cm)',y='petal width (cm)',color='Klasse:N').interactive()
    st.altair_chart(chart)

with tab3:
    st.latex(r'''
             Entropie = H =-\sum_{i=1}^n{p_i log_2(p_i)}
             ''')
    st.code(r""" def _calc_Entropy(self,y):
         '''Für jedes Feature wird die Entropie gemäß der Gleichung berechnet.'''
         H_ = []
         classes, counts = np.unique(y,return_counts=True)
     
         for i in range(len(classes)):
             p = counts[i]/sum(counts) #Berechnung der Wahrscheinlichkeitfür eine Klasse
             entropy = -p*np.log2(p) #Entropie Formel
             H_.append(entropy)
         return sum(H_)""")
     
    st.latex(r'''
              Information \; Gain = \Delta IG = H_{parent} - \frac{1}{N} \sum{N_{child} \cdot H_{child}}
              ''')
    st.code(r"""
            def _calc_information_gain(self,X_column,y,threshold):
                '''Hilfsfunktion. berechnet den Information gain für 2 Teilmengen einer Datenmenge, aufgeteilt an einer threshold, über der noch iteriert werden muss'''
                h_parent = self._calc_Entropy(y)
                #maske
                mask_r = X_column >= threshold
                mask_l = X_column < threshold
                n_ges = len(y)
                n_r = sum(mask_r)
                n_l = sum(mask_l)
                h_r = self._calc_Entropy(y[mask_r])
                h_l = self._calc_Entropy(y[mask_l])
                IG = h_parent -( ((n_r/n_ges)*h_r) + ((n_l/n_ges)*h_l)  )
                return IG""")

with tab4:
    ####
    column1, column2 = st.columns([3,1])
    
    
    with column2: #Threshold slider
        #----Mögliche thresholds, damit über jedes Objekt iteriert werden kann----#
        sorted_values_0 = np.sort(X.iloc[:,0].unique())
        sorted_values_1 = np.sort(X.iloc[:,1].unique())
        thresholds_0 = (sorted_values_0[1:] + sorted_values_0[:-1]) / 2 
        thresholds_1 = (sorted_values_1[1:] + sorted_values_1[:-1]) / 2 
        
        #---Achse0----#
        thres_0 = st.select_slider('**thresholds petal length**',list(thresholds_0)) # Alle Werte für die Achse 0
        IG_0 = _calc_information_gain(X.iloc[:,0], y, thres_0)
        st.write(f'Information Gain petal length: :red[**{IG_0:.4f}**]')
        st.divider()
        
        #---Achse1----#
        thres_1 = st.select_slider('**thresholds petal width**',list(thresholds_1)) # Alle Werte für die Achse 0
        IG_1 = _calc_information_gain(X.iloc[:,1], y, thres_1)
        st.write(f'Information Gain petal width: :blue[{IG_1:.4f}] ')
        
    with column1:
        #----Plot----#
        chart = alt.Chart(df_print).mark_circle().encode(x='petal length (cm)',y='petal width (cm)',color='Klasse:N').interactive()
        
        #----line-plots----#
        vertical_pos = thres_0
        horizontal_pos = thres_1
    
        vertical_line = alt.Chart(pd.DataFrame({'x': [vertical_pos]})).mark_rule(
            color='red').encode(x='x:Q')
        
        horizontal_line = alt.Chart(pd.DataFrame({'y':[horizontal_pos]})).mark_rule(color='blue').encode(y='y:Q')
        
        #----zusam,mengesetzter Chart----#
        final_Chart = (chart+vertical_line+horizontal_line).properties(
        width=700,
        height=450,
        title="Entscheidungsgrenzen").interactive()
        st.altair_chart(final_Chart)
    
    #----Max IG----#
    if thres_0 > thres_1:
        st.write(f'**max IG = {IG_0:.2f} auf x-Achse**')
    else:
        st.write(f'**max IG = {IG_1:.2f} auf y-Achse**')

with tab5: #Zeigt den Source Code
    st.code(inspect.getsource(DecisionTree),language='python')
    
with tab6:
    st.write('**X**')
    st.dataframe(X)
    st.write('**y**')
    st.dataframe(y)
    
with tab7:
    column1,column2 = st.columns(2)
    
    st.write('**Benötigter Code**')
    st.code('''
    Tree = DecisionTree()
    Tree.fit(X,y)
    ''')

    #----Methoden----#
    st.write('**Aufgerufene Methoden**')
    st.code(inspect.getsource(DecisionTree.fit),language='python')
    st.code(inspect.getsource(DecisionTree._build_tree))
    st.code(inspect.getsource(DecisionTree.Node),language='python')
    
    
with tab8:
    
    #----Entscheidungsbereiche plotten----#
    st.subheader("Entscheidungsgebiete des trainierten Entscheidungsbaums")
    
    #----Columns----#
    column1,column2 = st.columns([3,1])
    
    #----Baum Einstellungen----#
    with column2: 
        st.markdown("**Baum-Einstellungen**")
        max_depth = st.slider('Maximale Tiefe',1,10)
        min_samples = st.slider('Minimale Samples',0,5)
        
    #----Entscheidungsgrenzen Visualisierung----#
    with column1:
        tree = DecisionTree(max_depth=max_depth, min_samples=min_samples)
        tree.fit(X, y)
        
        #----Visualisierung----#
        
        #Meshgrid
        x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
        y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
        
        step = 0.1
        x_range = np.arange(x_min, x_max, step)
        y_range = np.arange(y_min, y_max, step)
        
        xx, yy = np.meshgrid(x_range, y_range)
        
        mesh_df = pd.DataFrame({
            X.columns[0]: xx.ravel(),
            X.columns[1]: yy.ravel()})
        
        #jeden Punkt im Meshgrid predicten
        mesh_df['Klasse_ID'] = tree.predict(mesh_df)
        mesh_df['Vorhersage'] = mesh_df['Klasse_ID'].map(dic) #Labels für Klassen erstellen
        
        #Farben für die Hintergrund Bereiche
        color_scale = alt.Scale(
            domain=['setosa', 'versicolor', 'virginica'],
            range=['#ff7f0e', '#1f77b4', '#2ca02c']) # Orange, Blau, Grün
        
        #Exakte Endpunkte für jedes Rechteck im Grid
        mesh_df['x2'] = mesh_df[X.columns[0]] + step
        mesh_df['y2'] = mesh_df[X.columns[1]] + step

        #Rechtecke
        background = alt.Chart(mesh_df).mark_rect(opacity=0.2).encode(
            x=alt.X(f'{X.columns[0]}:Q', title='Petal Length (cm)'),
            x2='x2',
            y=alt.Y(f'{X.columns[1]}:Q', title='Petal Width (cm)'),
            y2='y2',
            color=alt.Color('Vorhersage:N', scale=color_scale, legend=None),tooltip=alt.value(None))

        #dem Dataframe wird eine column mit dem vorhergesagten Werten angehängt (Zum Vergleich im plot)
        df_print['vorhersage'] = tree.predict(X)
        df_print['vorhersage'] = df_print['vorhersage'].map(dic) 
        
        #Punkt Chart mit den Objekten
        points = alt.Chart(df_print).mark_circle(size=60, stroke='white', strokeWidth=1).encode(
            x=alt.X('petal length (cm):Q'),
            y=alt.Y('petal width (cm):Q'),
            color=alt.Color('Klasse:N', scale=color_scale, title="Echte Klasse"),
            tooltip=['Klasse','vorhersage'])
        
        #Chart aus Punkten und background zusammensetzen
        final_chart = (background + points).properties(
            width=650,
            height=450,
            title=f"ID3 Entscheidungsgebiete (Tiefe: {max_depth})").interactive()
        
        #Chart visualisieren
        st.altair_chart(final_chart, use_container_width=True)
        
        
    st.divider()
    st.write('Neue Daten vorhersagen:')
    st.code('X: shape=(n_objects,n_features)')
    st.code('tree.predict(X)')
    st.divider()
    
    #----Vorhergesagte Daten anzeigen----#
    predicted = tree.predict(X)
    X_predicted = X.copy()
    X_predicted['predict'] = predicted
    st.write('Predicted:')
    st.dataframe(X_predicted)
    
with tab9:
    st.markdown("**Baum-Einstellungen**")
    max_depth_moons = st.slider('Maximale Tiefe Moons',1,10)
    min_samples_moons = st.slider('Minimale Samples Moons',0,30)
    
    
    noise_moons = st.slider('Noise',min_value=0.01,max_value=0.25,step=0.01)
    
    if 'randint' not in st.session_state:
        st.session_state.randint = random.randint(1,100)
    if st.button('reset random state'):
        st.session_state.randint = random.randint(1,100)
    
    
    #----MOONS importieren----#
    X_array, y_array = make_moons(n_samples=150,noise=noise_moons,random_state=st.session_state.randint)
    X_moons = pd.DataFrame(X_array,columns=['1','2'])
    y_moons = pd.DataFrame(y_array,columns=['target'])
    
    df_moons_ges= X_moons.copy()
    df_moons_ges['target'] = y_moons
    
    chart_moons = alt.Chart(df_moons_ges).mark_circle().encode(x='1',y='2',color='target:N').interactive()
    #st.altair_chart(chart_moons)
    
    y_moons = y_moons['target']
    
    Entscheidungsgrenzen(max_depth_moons,min_samples_moons,X_moons,y_moons)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


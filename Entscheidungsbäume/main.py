from sklearn.datasets import load_iris
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
from functions import _calc_information_gain, _calc_Entropy, DecisionTree



st.set_page_config(layout="wide")
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







tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8 = st.tabs(["Folie 1: Intro", "Folie 2: Iris Datensatz", "Folie 3: mathematische Grundlage","Folie 4: Visualisierung","Folie5: Code","Folie 6:Trainingsdaten","Folie 7: Training","Folie 8: Baum Visualisierung"])


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
    
    
    with column2:
        sorted_values_0 = np.sort(X.iloc[:,0].unique())
        sorted_values_1 = np.sort(X.iloc[:,1].unique())
        thresholds_0 = (sorted_values_0[1:] + sorted_values_0[:-1]) / 2 
        thresholds_1 = (sorted_values_1[1:] + sorted_values_1[:-1]) / 2 
        ####
        thres_0 = st.select_slider('**thresholds petal length**',list(thresholds_0)) # Alle Werte für die Achse 0
        IG_0 = _calc_information_gain(X.iloc[:,0], y, thres_0)
        st.write(f'Information Gain petal length: :red[**{IG_0:.4f}**]')
        
        st.divider()
        
        
        thres_1 = st.select_slider('**thresholds petal width**',list(thresholds_1)) # Alle Werte für die Achse 0
        IG_1 = _calc_information_gain(X.iloc[:,1], y, thres_1)
        st.write(f'Information Gain petal width: :blue[{IG_1:.4f}] ')
    with column1:
        #Plot
        chart = alt.Chart(df_print).mark_circle().encode(x='petal length (cm)',y='petal width (cm)',color='Klasse:N').interactive()
        
        
        vertical_pos = thres_0
        horizontal_pos = thres_1
    
        vertical_line = alt.Chart(pd.DataFrame({'x': [vertical_pos]})).mark_rule(
            color='red').encode(x='x:Q')
        
        horizontal_line = alt.Chart(pd.DataFrame({'y':[horizontal_pos]})).mark_rule(color='blue').encode(y='y:Q')
        
        final_Chart = (chart+vertical_line+horizontal_line).properties(
        width=700,
        height=450,
        title="Entscheidungsgrenzen").interactive()
        st.altair_chart(final_Chart)
    
    if thres_0 > thres_1:
        st.write(f'**max IG = {IG_0:.2f} auf x-Achse**')
    else:
        st.write(f'**max IG = {IG_1:.2f} auf y-Achse**')

with tab5:
    st.code("""
            class DecisionTree:
                '''
                Die Klasse basiert auf dem ID3 Algorithmus und kann zur Klassifikation von Daten mit kontinuierlichen Werten anhand von 2 Features verwendet werden
                Die fit Methode bekommt ein pandas Dataframe X mit 2 features in den beiden Spalten und ein Vektor y mit den Zielwerrten für das Training.
                Die predict Methode liefert für ein Dataframe X ein Vektor mit den vorhergesagten Klassen.
                x
                Parameter
                --------
                max_depth: int, default=3
                    Die maximale Baumtiefe. Ein zu großer Wert fürt zu overfitting
                min_samples: int, default=2
                    Minimale Anzahl an Objekten für einen Node
                
                Attribute
                --------
                root: Node object
                    Definiert durch die fit Methode. Enthält den kompletten Entscheidungsbaum angefangen mit der ersten Node,
                    welche auf die weteren verweist.'''
                def __init__(self,depth=0,max_depth=3,min_samples=2):
                    self.depth = depth
                    self.max_depth = max_depth
                    self.min_samples = min_samples
                    self.min_samples = min_samples
                class Node:
                    '''
                    Bauplan für die Entscheidungspunkte des Baumes.
                    Attribute
                    --------
                    feature: int, None bei einem Blatt
                    feature: int, value bei einem Entscheidungsknoten'''
                    def __init__(self,feature=None,threshold=None,left_child=None,right_child=None,depth=None,axis=None):
                        self.feature = feature
                        self.threshold = threshold
                        self.left_child= left_child
                        self.right_child = right_child
                        self.depth = depth
                        self.axis = axis

                def fit(self,X,y):
                    '''
                    Trainiert den Entscheidungsbaum mit den Trainingsdaten.

                    Paramter
                    --------
                    X: DataFrame, shape = (n_objects,n_features)
                    y: DataFrame, shape= (n_objects,)
                    
                    Returns
                    --------
                    None 
                        Der Entscheidungsbaum wird in der Klasse in self.root gespeichert
                    '''
                    self.root = self._build_tree(X, y, depth=0)
                
                def predict(self,X):
                    '''
                    Klassifiziert ein Dataframe.
                    Parameter
                    --------
                    X: DataFrame, shape=(n_objects,n_features)
                    
                    Returns
                    --------
                    Liste mit Klassifizierung der Objekte'''
                    X_array = np.array(X)

                    pred_list = []
                    for x in X_array:
                        pred = self._check_object(x,self.root)
                        pred_list.append(pred)
                    
                    return pred_list

                    
                def _build_tree(self,X,y,depth=0):   
                    '''Diese Methode beeinhaltet den eigentlichen Algorithmus.
                    Zuerst wird überprüft, ob eine der Abbruchbedingungen erreicht wird. Falls ja, wird geschaut, welche Klasse am häufigsten vorkommt 
                    und anschließend wird ein Leaf-Node erstellt mit dieser Zielklasse.
                    
                    Ist keine Abbruchbedingung erfüllt, muss eine Entscheidungsgrenze gefunden werden. 
                    Dafür wird, iteriert über jeden Datenpunkt(Über beide Features), der Information Gain berechnet. 
                    Aus dem entstanden Schwellenwert (threshold) wird dann eine Maske (bool-array) erstellt.
                    
                    Beide entstanden Teilmengen werden rekursiv der _build_tree Methode gegeben
                    Daraus kann dann wieder eintweder ein leaf oder 2 weitere Kinder (childs) entstehen.
                    
                    Beide Objekte werden dann als current_Node gespeichert.'''
                    #print('depth=',depth)
                    if depth >= self.max_depth or len(X) <= self.min_samples or int(len(y.unique())) == 1:
                        value = int(y.value_counts().idxmax())
                        return self.Node(feature=value) #Blatt wird generiert

                    thresh, axis_ = self._calc_max_IG(X,y)
                    mask_ = self._mask(X,thresh,axis_)

                    
                    left_child_ = self._build_tree(X[mask_],y[mask_],depth+1)
                    right_child_ = self._build_tree(X[~mask_],y[~mask_],depth+1)

                    
                    current_Node = self.Node(threshold=thresh,left_child=left_child_,right_child=right_child_,axis=axis_)
                    return current_Node
                    
                def _calc_Entropy(self,y):
                    '''Für jedes Feature wird die Entropie gemäß der Gleichung berechnet.'''
                    H_ = []
                    classes, counts = np.unique(y,return_counts=True)
                
                    for i in range(len(classes)):
                        p = counts[i]/sum(counts) #Berechnung der Wahrscheinlichkeitfür eine Klasse
                        entropy = -p*np.log2(p) #Entropie Formel
                        H_.append(entropy)
                    return sum(H_)

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
                    return IG

                def _calc_max_IG_column(self,X_column,y):
                    '''berechnet den Information gain für jeden möglichen Grenzwert (threshold) und gibt den höchsten Information gain und den dazu passenden Schwellwert aus.'''
                    sorted_values = np.sort(X_column.unique())
                    thresholds = (sorted_values[1:] + sorted_values[:-1]) / 2 
                    #print(thresholds)
                    
                    best_IG = -1
                    best_IG_threshold = 0
                    
                    for i in thresholds:
                        current_IG = self._calc_information_gain(X_column,y,i)  
                        if current_IG > best_IG:
                            best_IG = current_IG
                            best_IG_threshold = i
                
                    return best_IG, best_IG_threshold


                def _calc_max_IG(self,X,y):
                    '''Berechnet den maximalen Information gain für beide features und gibt den Schwellwert und das jeweilige Feature(axis) aus'''
                    X_columns_1 = X.iloc[:,0]
                    X_columns_2 = X.iloc[:,1]
                    max_IG_1, threshold_1 = self._calc_max_IG_column(X_columns_1,y)
                    max_IG_2, threshold_2 = self._calc_max_IG_column(X_columns_2,y)
                    
                
                    
                    if max_IG_1 > max_IG_2:
                        return threshold_1, 0
                    else:
                        return threshold_2, 1
                
                def _mask(self,X,threshold,axis):
                    '''Ertellt eine Maske (bool-array) um die Teilmengen zu erstellen'''
                    bool_array = X.iloc[:,axis] <= threshold
                    return bool_array
                        
                def _check_object(self,x,node):
                    '''geht die Nodes durch und gibt den passenden,vorhergesagten, Klassenwert aus'''
                    #Leaf-Check
                    if node.left_child == None and node.right_child == None:
                        return node.feature
                    
                    #gesuchter x wert, je nach axis
                    _x = x[node.axis]

                    if _x <= node.threshold:
                        return self._check_object(x,node.left_child)
                        
                    else:
                        return self._check_object(x,node.right_child)""")
    
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

    st.write('**Aufgerufene Methoden**')
    st.code("""
            def fit(self,X,y):
                '''
                Trainiert den Entscheidungsbaum mit den Trainingsdaten.

                Paramter
                --------
                X: DataFrame, shape = (n_objects,n_features)
                y: DataFrame, shape= (n_objects,)
                
                Returns
                --------
                None 
                    Der Entscheidungsbaum wird in der Klasse in self.root gespeichert
                '''
                self.root = self._build_tree(X, y, depth=0)""")
    st.code("""
            def _build_tree(self,X,y,depth=0):   
                '''Diese Methode beeinhaltet den eigentlichen Algorithmus.
                Zuerst wird überprüft, ob eine der Abbruchbedingungen erreicht wird. Falls ja, wird geschaut, welche Klasse am häufigsten vorkommt 
                und anschließend wird ein Leaf-Node erstellt mit dieser Zielklasse.
                
                Ist keine Abbruchbedingung erfüllt, muss eine Entscheidungsgrenze gefunden werden. 
                Dafür wird, iteriert über jeden Datenpunkt(Über beide Features), der Information Gain berechnet. 
                Aus dem entstanden Schwellenwert (threshold) wird dann eine Maske (bool-array) erstellt.
                
                Beide entstanden Teilmengen werden rekursiv der _build_tree Methode gegeben
                Daraus kann dann wieder eintweder ein leaf oder 2 weitere Kinder (childs) entstehen.
                
                Beide Objekte werden dann als current_Node gespeichert.'''
                #print('depth=',depth)
                if depth >= self.max_depth or len(X) <= self.min_samples or int(len(y.unique())) == 1:
                    value = int(y.value_counts().idxmax())
                    return self.Node(feature=value) #Blatt wird generiert

                thresh, axis_ = self._calc_max_IG(X,y)
                mask_ = self._mask(X,thresh,axis_)

                
                left_child_ = self._build_tree(X[mask_],y[mask_],depth+1)
                right_child_ = self._build_tree(X[~mask_],y[~mask_],depth+1)

                
                current_Node = self.Node(threshold=thresh,left_child=left_child_,right_child=right_child_,axis=axis_)
                return current_Node""")
    st.code("""class Node:
        '''
        Bauplan für die Entscheidungspunkte des Baumes.
        Attribute
        --------
        feature: int, None bei einem Blatt
        feature: int, value bei einem Entscheidungsknoten'''
        def __init__(self,feature=None,threshold=None,left_child=None,right_child=None,depth=None,axis=None):
            self.feature = feature
            self.threshold = threshold
            self.left_child= left_child
            self.right_child = right_child
            self.depth = depth
            self.axis = axis""")
    
    
with tab8:
    ###Entscheidungsbereiche plotten###
    st.subheader("Entscheidungsgebiete des trainierten Entscheidungsbaums")
    column1,column2 = st.columns([3,1])
    with column2:
        st.markdown("**Baum-Einstellungen**")
        max_depth = st.slider('Maximale Tiefe',1,10)
        min_samples = st.slider('Minimale Samples',0,5)
    with column1:
        tree = DecisionTree(max_depth=max_depth, min_samples=min_samples)
        tree.fit(X, y)
        
        ### Visualisierung ###
        x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
        y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
        
        step = 0.1
        x_range = np.arange(x_min, x_max, step)
        y_range = np.arange(y_min, y_max, step)
        
        xx, yy = np.meshgrid(x_range, y_range)
        
        mesh_df = pd.DataFrame({
            X.columns[0]: xx.ravel(),
            X.columns[1]: yy.ravel()
        })
        
        mesh_df['Klasse_ID'] = tree.predict(mesh_df)
        mesh_df['Vorhersage'] = mesh_df['Klasse_ID'].map(dic)
        
        color_scale = alt.Scale(
            domain=['setosa', 'versicolor', 'virginica'],
            range=['#ff7f0e', '#1f77b4', '#2ca02c']) # Orange, Blau, Grün
        
        
        # 1. Definiere die exakten Endpunkte für jedes Rechteck im Grid
        mesh_df['x2'] = mesh_df[X.columns[0]] + step
        mesh_df['y2'] = mesh_df[X.columns[1]] + step

        # 2. Zeichne die Rechtecke mit expliziten Koordinaten (ohne alt.Bin!)
        background = alt.Chart(mesh_df).mark_rect(opacity=0.2).encode(
            x=alt.X(f'{X.columns[0]}:Q', title='Petal Length (cm)'),
            x2='x2',
            y=alt.Y(f'{X.columns[1]}:Q', title='Petal Width (cm)'),
            y2='y2',
            color=alt.Color('Vorhersage:N', scale=color_scale, legend=None),tooltip=alt.value(None))

        
        df_print['vorhersage'] = tree.predict(X)
        df_print['vorhersage'] = df_print['vorhersage'].map(dic) 
        points = alt.Chart(df_print).mark_circle(size=60, stroke='white', strokeWidth=1).encode(
            x=alt.X('petal length (cm):Q'),
            y=alt.Y('petal width (cm):Q'),
            color=alt.Color('Klasse:N', scale=color_scale, title="Echte Klasse"),
            tooltip=['Klasse','vorhersage']
        )
        
        final_chart = (background + points).properties(
            width=650,
            height=450,
            title=f"ID3 Entscheidungsgebiete (Tiefe: {max_depth})"
        ).interactive()
        
        st.altair_chart(final_chart, use_container_width=True)
        ######################
    
    predicted = tree.predict(X)
    X_predicted = X.copy()
    X_predicted['predict'] = predicted
    st.dataframe(X_predicted)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


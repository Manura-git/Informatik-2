from sklearn.datasets import load_iris
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st


class DecisionTree:
    '''
    Die Klasse basiert auf dem ID3 Algorithmus und kann zur Klassifikation von Daten mit kontinuierlichen Werten anhand von 2 Features verwendet werden
    Die fit Methode bekommt ein pandas Dataframe X mit 2 features in den beiden Spalten und ein Vektor y mit den Zielwerrten für das Training.
    Die predict Methode liefert für ein Dataframe X eine Series mit den vorhergesagten Klassen.
    
    Parameter
    --------
    max_depth: int, default=3
        Die maximale Baumtiefe. Ein zu großer Wert fürt zu overfitting
    min_samples: int, default=2
        Minimale Anzahl an Objekten für einen Node
    
    Attribute
    --------
    root: Node object
        Definiert durch die fit Methode. Enthält den kompletten Entscheidungsbaum 
        '''
    def __init__(self,depth=0,max_depth=3,min_samples=2):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_samples = min_samples
    class Node:
        '''
        Bauplan für die Entscheidungspunkte und Blätter des Baums.
        
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
        global_default = int(y.value_counts().idxmax()) if len(y) > 0 else None #Die häufigste Klasse des gesamten Datensatzes als Sicherheit
        
        self.root = self._build_tree(X, y, depth=0,parent_default=global_default)
    
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

        
    def _build_tree(self,X,y,depth=0,parent_default=None):   
        '''
        Die Methode enthält den Algorithmus. Sie nimmt die Trainingsdaten und erstellt Entschiedungs- und Blattknoten.
        Gibt es mehr als 0 Objekte in der Teilmenge wird die am häufigsen vorkommende Klasse in current_default gespeichert. Bei einer Leeren Menge wird auf das Eltern Value verwiesen.
        Wird eine der Abbruchbedingungen erreicht bricht der Algorithmus hier ab und erstellt einen leaf-Knoten.
        Anderenfalls wird die Menge in 2 Teilmengen (lnks und rechts) aufgeteilt und der Vorgang so lange rekursiv wiederholt bis man am Ende immer auf einen Leaf-Knoten trifft.
        
        returns
        --------
        Node Objekt
        
        '''
        #----Value definieren----#
        if len(y) > 0:
            current_default = int(y.value_counts().idxmax()) #Der Wert der am Häufigsten vorkommt wird in current_default gespeichert
        else:
            current_default = parent_default #Ist die Teilmenge leer, wird die Klassifikation von dem parent-Node übernommen
        
        #obsolet
        #if len(X) == 0 or len(y) == 0:
        #    return self.Node(feature=parent_default) #Wenn X oder y leer sind, wird nicht None zurückgegeben sondern die Klassifizierung des Elternknotens
        
        #----Abbruchbedingungen----#
        if depth >= self.max_depth or len(X) <= self.min_samples or int(len(y.unique())) == 1: 
            return self.Node(feature=current_default) #Erstellt leaf

        #----Entscheidungsgrenze----#
        thresh, axis_ = self._calc_max_IG(X,y) #Die Entscheidungsgrenze und die betrachtete Achse (mit dem max. IG) 
        mask_ = self._mask(X,thresh,axis_) #Bool Array; trennt linken und rechten Bereich
        
        #----Zusätzliche Abbruchbedingung----#  Wenn es nur noch eine mögliche Klasse gibt wird der Baum abgebrochen
        if len(X.iloc[:, axis_].unique()) == 1:
            return self.Node(feature=current_default) #Erstellt leaf
        
        #----Rekursion----#
        left_child_ = self._build_tree(X[mask_],y[mask_],depth+1,parent_default=current_default) #Gibt den linken Bereich rekursiv an _build_tree
        right_child_ = self._build_tree(X[~mask_],y[~mask_],depth+1,parent_default=current_default) #Durch die invertierte Maske wird der rechte Bereich an _build_tree zurückgegeben

        current_Node = self.Node(feature=current_default,threshold=thresh,left_child=left_child_,right_child=right_child_,axis=axis_)
        return current_Node #Erstellt Entscheidungsknoten Dieser enthält Verweise weitere Knoten
        
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
        
    def _calc_max_IG_column(self,X_column,y):
        '''berechnet den Information gain für jeden möglichen Grenzwert (threshold) und gibt den höchsten Information gain und den dazu passenden Schwellwert aus.'''
        #----mögliche Thresholds berechnen so, dass einmal über jedes Objekt iteriert werden kann----#
        sorted_values = np.sort(X_column.unique())
        thresholds = (sorted_values[1:] + sorted_values[:-1]) / 2 
        
        best_IG = -1
        best_IG_threshold = 0
        
        for i in thresholds:
            current_IG = self._calc_information_gain(X_column,y,i)  
            if current_IG > best_IG:
                best_IG = current_IG
                best_IG_threshold = i
    
        return best_IG, best_IG_threshold
        
    def _calc_information_gain(self,X_column,y,threshold):
        '''Hilfsfunktion. berechnet den Information gain für 2 Teilmengen einer Datenmenge, aufgeteilt an einer threshold, über der noch iteriert werden muss'''
        h_parent = self._calc_Entropy(y)
        #maske
        mask_r = X_column > threshold
        mask_l = X_column <= threshold
        n_ges = len(y)
        n_r = sum(mask_r) #Menge der Objekte im linke child
        n_l = sum(mask_l) #menge der Objekte im rechten child
        h_r = self._calc_Entropy(y[mask_r]) #Entropie für das rechte child
        h_l = self._calc_Entropy(y[mask_l]) #Entropie für das rechte child
        IG = h_parent -( ((n_r/n_ges)*h_r) + ((n_l/n_ges)*h_l)  )
        return IG
    
    def _calc_Entropy(self,y):
        '''Für jedes Feature wird die Entropie gemäß der Gleichung berechnet.'''
        H_ = []
        classes, counts = np.unique(y,return_counts=True) #Klassen und Mengenverteilung für die Berechnung
    
        for i in range(len(classes)):
            p = counts[i]/sum(counts) #Berechnung der Wahrscheinlichkeit für eine Klasse
            entropy = -p*np.log2(p) #Entropie Formel
            H_.append(entropy)
        return sum(H_)

    def _mask(self,X,threshold,axis):
        '''Ertellt eine Maske (bool-array) um die Teilmengen zu erstellen'''
        bool_array = X.iloc[:,axis] <= threshold
        return bool_array
            
    def _check_object(self,x,node):
        '''geht die Nodes durch und gibt den passenden,vorhergesagten, Klassenwert aus'''
        #----Sicherheits-Abbruch----#
        if node is None:
            return None
        
        #----Leaf-Check----#
        if node.left_child is None and node.right_child is None:
            return node.feature
        
        _x = x[node.axis] #referenziert das für den Knoten relevante feature in _x

        #----Rekursion----#
        #An diesem Punkt ist es klar, dass es sich um einen Entscheidungsknoten handelt(der Leaf-check wurde nicht ausgelöst)
        #Der Algorithmus führt dann die _check_object() Methode auf einem der Kinder aus. Das wird so lange wiederholt
        #bis der Algorithmus auf einen Leaf-Knoten trifft und die im Knoten gespeicherte Value wird ausgegeben
        if _x <= node.threshold:
            if node.left_child is not None:
                return self._check_object(x,node.left_child)
            else:
                return node.feature
        else:
            if node.right_child is not None:
                return self._check_object(x,node.right_child)
            else:
                return node.feature
    
    
    
    
    
    
    
#Methoden als eigene Funktionen nur für die Visualisierung in Streamlit
def _calc_Entropy(y):
    '''Für jedes Feature wird die Entropie gemäß der Gleichung berechnet.'''
    H_ = []
    classes, counts = np.unique(y,return_counts=True)

    for i in range(len(classes)):
        p = counts[i]/sum(counts) #Berechnung der Wahrscheinlichkeitfür eine Klasse
        entropy = -p*np.log2(p) #Entropie Formel
        H_.append(entropy)
    return sum(H_)
def _calc_information_gain(X_column,y,threshold):
    '''Hilfsfunktion. berechnet den Information gain für 2 Teilmengen einer Datenmenge, aufgeteilt an einer threshold, über der noch iteriert werden muss'''
    h_parent = _calc_Entropy(y)
    #maske
    mask_r = X_column >= threshold
    mask_l = X_column < threshold
    n_ges = len(y)
    n_r = sum(mask_r)
    n_l = sum(mask_l)
    h_r = _calc_Entropy(y[mask_r])
    h_l = _calc_Entropy(y[mask_l])
    IG = h_parent -( ((n_r/n_ges)*h_r) + ((n_l/n_ges)*h_l)  )
    return IG
class Node:
    '''
    Bauplan für die Entscheidungspunkte und Blätter des Baums.
    
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







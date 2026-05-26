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
            entropy = -p*np.log2(p) #Entrope Formel
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
            return self._check_object(x,node.right_child)
    

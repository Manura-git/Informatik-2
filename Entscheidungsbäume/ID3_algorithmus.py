from sklearn.datasets import load_iris
import altair as alt
import pandas as pd
import numpy as np



df = load_iris()
target = df.target
df = df.data[:,2:] #only petal width and height

df = pd.DataFrame(df,columns=['petal length','petal width'])
df['target'] = target
X = df

chart = alt.Chart(X).mark_circle().encode(x='petal length',y='petal width',color='target:N')
alt.renderers.enable('browser')
chart.show()

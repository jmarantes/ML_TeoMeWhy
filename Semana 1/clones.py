#%%

import pandas as pd

df = pd.read_parquet('data/dados_clones.parquet')
df

#%%
features = ['Massa(em kilos)', 'Estatura(cm)', 
            'Distância Ombro a ombro', 'Tamanho do crânio',
            'Tamanho dos pés']
target = ['Status ']

X = df[features]
y = df[target]

X = X.replace(
    {
        "Tipo 1" : "1",
        "Tipo 2" : "2",
        "Tipo 3" : "3",
        "Tipo 4" : "4",
        "Tipo 5" : "5"
    }
)

X
# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)
# %%
import matplotlib.pyplot as plt

plt.figure(dpi = 600)
tree.plot_tree(model, 
               feature_names=features,
               class_names=model.classes_,
               filled=True,
               max_depth=3)
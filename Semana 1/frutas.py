# %%

import pandas as pd

df = pd.read_excel('data/dados_frutas.xlsx')
df

# %%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42) #random_state define o seed e tira a aleatoriedade

#%%

y = df['Fruta']

caracteristicas = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']
x = df[caracteristicas]

#%%
# AQUI É MACHINE LEARNING!! ALGORITMO DE ÁRVORE
arvore.fit(x, y)

#%%

arvore.predict([[0,0,0,0]])
#%%

import matplotlib.pyplot as plt
#%%

plt.figure(dpi=400)
tree.plot_tree(arvore, 
               feature_names=caracteristicas,
               class_names=arvore.classes_,
               filled=True
               )
#%%

proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)
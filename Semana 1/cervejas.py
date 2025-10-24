#%%

import pandas as pd

df = pd.read_excel('data/dados_cerveja.xlsx')
df.head()
#%%

features = ['temperatura', 'copo', 'espuma', 'cor']
target = ['classe']

X = df[features]
y = df[target]

X = X.replace(
    {
        "mud" : 1, "pint" : 2,
        "sim" : 1, "não" : 0,
        "clara" : 0, "escura" : 1
    }
)

#%%

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)

# Para os modelos do SKLEARN, todas as variáveis precisam ser do tipo numérico
#%%
import matplotlib.pyplot as plt

plt.figure(dpi = 600)
tree.plot_tree(model, 
               feature_names=features,
               class_names=model.classes_,
               filled=True)
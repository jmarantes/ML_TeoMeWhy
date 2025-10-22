# %%

import pandas as pd

# %%

df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df
# %%
from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']]
y = df['nota']

# Ajustamos o modelo (MACHINE LEARNING)

reg = linear_model.LinearRegression()
reg.fit(X, y)

# Exibindo o coeficiente
a, b = reg.intercept_, reg.coef_[0]
print(a, b)

# Novas predições com base nos dados
predict_reg = reg.predict(X.drop_duplicates())

# Fazendo o processo com a Árvore de Decisão ajusta 100% pelos dados 
# Com isso, temos um overfit. A árvore está super ajustada aos meus dados
arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)

predict_arvore_full = arvore_full.predict(X.drop_duplicates())

# Alteramos o hiper_parâmetro max_depth, para impedir que a árvore cresca 
# até chegar em uma única amostra por nó, evitando o overfit
# A árvore possui alguns critérios de parada, como o max_depth que utilizamos
arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X, y)

predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
# %%

import matplotlib.pyplot as plt

# Pontos plotados
plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relação Cerveja x Nota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

# Predição da regressão linear plotada
plt.plot(X.drop_duplicates()['cerveja'], predict_reg)
# Árvore com overfit plotada
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full, color='green')
# Árvore plotada
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2, color='magenta')

plt.legend(['Observado', f'y = {a:.3f} + {b:.3f}x', 'Árvore Full', 'Árvore D2'])

# %%

tree.plot_tree(arvore_d2, feature_names=['ceveja'], filled=True)
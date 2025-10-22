#%%

import pandas as pd

url = 'https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbV9MVGRnSlJPN2RwbFVkYU43UzZMVlItdE9Vd3xBQ3Jtc0tuSlhBX1VDN3pEcnhkdXQ1b2J2NkN2SzVPYWt6TkhibElHa2V6c0t6akRJSjE5WDdieV9HbTJRMkktcHU5ci1xTTNHUzAzNTNiOHBlODZHaXdYQ0I2T2dYc2dRZHZaM0JibjRKV0htZ09CTHpTM2h5NA&q=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2F1YQBQ3bu1TCmgrRch1gzW5O4Jgc8huzUSr7VUkxg0KIw%2Fexport%3Fgid%3D283387421%26format%3Dcsv&v=ImWgtWmP61s'

df = pd.read_csv('data/dados_comunidade.csv')
df.head()

#%%

df = df.replace(
    {
    'Sim' : 1,
    'Não' : 0
})

df

#%%

num_vars = ['Curte games?',
            'Curte futebol?', 'Curte livros?', 
            'Curte jogos de tabuleiro?', 'Curte jogos de fórmula 1?', 
            'Curte jogos de MMA?', 'Idade',
]

dummy_vars = ['Como conheceu o Téo Me Why?',
              'Quantos cursos acompanhou do Téo Me Why?',
              'Estado que mora atualmente',
              'Área de Formação',
              'Tempo que atua na área de dados', 
              'Posição da cadeira (senioridade)',
            ]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise['pessoa feliz'] = df['Você se considera uma pessoa feliz?']
df_analise

# ATÉ AQUI! Apenas transformamos os dados em numéricos
# Criamos uma ABT (Analytical Base Table)

#%%

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42,
                                     min_samples_leaf=5
                                     )

features = df_analise.columns[:-1].tolist() # Da primeira até a penúltima coluna
X = df_analise[features]
y = df_analise['pessoa feliz'] # Variável resposta na última coluna

arvore.fit(X, y)

#%%

arvore_predict = arvore.predict(X)
arvore_predict

df_predict = df_analise[['pessoa feliz']]
df_predict['predict_arvore'] = arvore_predict
df_predict

#%%

# Acurácia !!!
(df_predict['pessoa feliz'] == df_predict['predict_arvore']).mean()

# A acurácia NÃO É SUFICIENTE!!!
# Ela nos informa o quanto estamos acertando, mas não mostra ONDE estamos acertando

#%%

# Matriz de Confusão
pd.crosstab(df_predict['predict_arvore'], df_predict['pessoa feliz'])

# Cruzamos quantas valores existem no observado e quantos existem na predição
# Assim vemos onde estamos acertando e errando mais

#%%

(df_predict['predict_arvore'] == 0).sum()
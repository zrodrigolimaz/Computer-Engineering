# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carregando os dados
df = pd.read_csv('Heart_Disease_Prediction.csv')

# Verificando se existem valores faltantes
print(df.isnull().sum())

# Pré-processamento: convertendo os rótulos de texto para numéricos
le = LabelEncoder()
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])

# Normalizando os dados
scaler = StandardScaler()
columns_to_scale = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Verificando a correlação entre as variáveis
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.show()

# Dividindo os dados em recursos (X) e alvo (y)
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliando o modelo
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f'Acurácia: {accuracy}')
print(f'Matriz de Confusão: \n{confusion}')

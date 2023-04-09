# Predição de Diabetes

Este projeto tem como objetivo criar um modelo de aprendizado de máquina para prever a ocorrência de diabetes em pacientes com base em um conjunto de características clínicas. O modelo utiliza a biblioteca scikit-learn e Python para realizar a análise e a predição.

## Conjunto de Dados

O conjunto de dados usado neste projeto está disponível no seguinte link:

https://docs.google.com/spreadsheets/d/1gBtv4l_xjSFnSB3Q8L7l0Hm0ynioN5d5z_jWV0yfIDk/edit#gid=381546931

Ele contém informações sobre diversos pacientes, como idade, gênero, pressão arterial e outras características clínicas, além de um rótulo binário que indica se o paciente tem diabetes (1) ou não (0).

## Dependências

As bibliotecas necessárias para executar este projeto estão listadas no arquivo `requirements.txt`. Para instalá-las, execute o seguinte comando:

pip install -r requirements.txt

## Como executar

Para executar o projeto, basta executar o arquivo `diabetes_prediction.py`:

python diabetes_prediction.py

O script irá carregar os dados, treinar um modelo de Regressão Logística e avaliar sua acurácia, matriz de confusão e relatório de classificação.

## Melhorias Futuras

Algumas melhorias possíveis para este projeto incluem:

- Exploração e visualização dos dados.
- Aplicação de validação cruzada e ajuste de hiperparâmetros.
- Teste de outros algoritmos de aprendizado de máquina e comparação de desempenho.
- Criação de um Jupyter Notebook para apresentação interativa do projeto.

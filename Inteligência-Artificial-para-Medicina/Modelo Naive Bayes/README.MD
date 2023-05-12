Análise de Doença Cardíaca com Naive Bayes
==========================================

Este repositório contém um script Python para a análise de dados de doenças cardíacas e treinamento de um modelo de Aprendizado de Máquina utilizando o algoritmo Naive Bayes.

Dependências
------------

O script depende das seguintes bibliotecas Python:

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   sklearn

Você pode instalar todas as dependências usando pip:


`pip install pandas numpy matplotlib seaborn sklearn`

Dados
-----

O script usa um conjunto de dados de doenças cardíacas, que você precisa fornecer na forma de um arquivo CSV chamado `Heart_Disease_Prediction.csv`. Este arquivo deve estar na mesma pasta do script Python para que ele funcione corretamente.

Como executar
-------------

Para executar o script, certifique-se de ter todas as dependências instaladas e os dados disponíveis, então execute o arquivo Python com seu interpretador Python de escolha.

O script irá carregar os dados, fazer alguma pré-processação (incluindo codificação de rótulos e normalização), treinar o modelo Naive Bayes, fazer previsões em um conjunto de testes e, em seguida, imprimir a acurácia do modelo e a matriz de confusão.

Sobre o algoritmo
-----------------

Naive Bayes é um algoritmo de Aprendizado de Máquina baseado no teorema de Bayes. Ele assume que a presença de um recurso particular em uma classe é independente da presença de qualquer outro recurso, mesmo que esses recursos dependam uns dos outros. Essa suposição simplificadora permite que o algoritmo seja treinado de maneira eficiente, embora sua performance possa ser afetada se as variáveis de entrada são dependentes.

Sobre o conjunto de dados
-------------------------

O conjunto de dados de doenças cardíacas consiste em várias variáveis relacionadas à saúde do coração, como idade, sexo, tipo de dor no peito, pressão arterial, colesterol, e muito mais. Estes são usados como recursos para treinar o modelo. O alvo é uma variável binária indicando se a pessoa tem ou não doença cardíaca.

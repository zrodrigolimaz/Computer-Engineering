import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(url):
    df = pd.read_csv(url)
    return df


def preprocess_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return accuracy, conf_matrix, class_report


def main():
    url = "https://docs.google.com/spreadsheets/d/1gBtv4l_xjSFnSB3Q8L7l0Hm0ynioN5d5z_jWV0yfIDk/export?format=csv"
    df = load_data(url)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    accuracy, conf_matrix, class_report = evaluate_model(model, X_test, y_test)

    print(f"Acurácia do modelo: {accuracy*100:.2f}%")
    print("Matriz de Confusão:")
    print(conf_matrix)
    print("Relatório de Classificação:")
    print(class_report)


if __name__ == "__main__":
    main()

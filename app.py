import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# URL do conjunto de dados Heart Disease do UCI ML Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

@st.cache_data()
def load_data():
    dados = pd.read_csv(url, names=column_names)
    
    # Tratamento de valores ausentes
    dados.replace('?', np.nan, inplace=True)
    dados = dados.apply(pd.to_numeric, errors='coerce')
    dados.dropna(inplace=True)
    
    return dados

@st.cache_data()
def train_model(X_train_std, y_train):
    # Construir o modelo MLP
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=128, input_dim=X_train_std.shape[1], activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treinar o modelo
    history = model.fit(X_train_std, y_train, validation_split=0.21, epochs=100, batch_size=32, verbose=1)
    
    return model, history

# Carregar dados
dados = load_data()

# Separar features e targets
X = dados.iloc[:, 0:14].values
y = dados['target'].values

# Dividir dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Treinar o modelo
model, history = train_model(X_train_std, y_train)

# Função para exibir a perda durante o treinamento
def exibePerda(fig):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    st.pyplot()

# Novo conjunto de dados de teste
novo_teste = np.array([[45.0, 1.0, 4.0, 160.0, 225.0, 0.0, 2.0, 156.0, 1.0, 3.3, 3.0, 0, 0, 0]])

# Previsão usando o modelo
saida_predita = model.predict(novo_teste)

# Se for uma saída binária (camada de saída com ativação sigmoid)
# Converta a saída para 0 ou 1 usando um limiar (por exemplo, 0.5)
saida_binaria = (saida_predita > 0.5).astype(int)

print(saida_binaria)

# Testar o modelo
y_pred = np.argmax(model.predict(X_test_std), axis=-1)

# Função para exibir a matriz de confusão e a precisão do modelo
def display_results(fig):
    y_pred = np.argmax(model.predict(X_test_std), axis=-1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    st.pyplot()
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Model Accuracy: {acc}')

# Testar o modelo
def exibiracuracia(fig):
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Model Accuracy: {acc}')
    plt.scatter(y_test, model.predict(X_test_std), alpha=0.5)
    plt.title('Previsões vs. Observações Reais')
    plt.xlabel('Observações Reais')
    plt.ylabel('Previsões do Modelo')
    st.pyplot()

# Função para exibir um histograma das probabilidades previstas
def histogrema(fig):
    plt.hist(model.predict(X_test_std), bins=20, edgecolor='black')
    plt.title('Histograma das Probabilidades Previstas')
    plt.xlabel('Probabilidades Previstas')
    plt.ylabel('Frequência')
    st.pyplot()

#-------------------------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.image(
    "https://cdn.sanity.io/images/tlr8oxjg/production/ada93729daf922ad0318c8c0295e5cb477921808-1456x816.png?w=3840&q=100&fit=clip&auto=format"
)
st.sidebar.header("Redes Neurais")
st.sidebar.title("Heart Disease Prediction App")

info = """
Este aplicativo foi desenvolvido para ajudar na previsão de doenças cardíacas com base em 
dados clínicos. Utilizando técnicas de aprendizado de máquina, o aplicativo analisa informações 
como idade, sexo, pressão sanguínea, colesterol e outros fatores relevantes para determinar a 
probabilidade de uma pessoa ter doença cardíaca.
"""

st.sidebar.info(info)


st.title("Heart Disease Prediction :red[Prediction] :bar_chart: :chart_with_upwards_trend:  :heart:")
st.markdown("")

tab1, tab2 = st.tabs(["Data :clipboard:", "Performance :bar_chart:"])
fig, ax = plt.subplots(figsize=(4, 4), dpi=10)
st.set_option('deprecation.showPyplotGlobalUse', False)


with tab1:
    data_df = dados.head()
    st.header("Dataset")
    st.write(data_df)
with tab2:
    st.header("Confusion Matrix | Feature Importances")
    col1, col2 = st.columns(2)
    
    with col1:
        
        st.subheader("Training Loss")
        exibePerda(fig)
        st.subheader("Confusion Matrix")
        display_results(fig)
        st.subheader("Model Accuracy")
        exibiracuracia(fig)
        st.subheader("Histogram of Predicted Probabilities")
        histogrema(fig)

    
   

        
        
        
        
        
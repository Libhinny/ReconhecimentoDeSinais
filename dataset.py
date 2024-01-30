# biblotecas para o uso de dados - bibliotecas auxiliares
import pandas as pd
import numpy as np
# tratativa de dados
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

 
print(tf._version_)


# caminho do dataset
dados = "C:\Users\libhi\Downloads\dataset_SignLanguage.zip"
treino = tf.keras.utils.get_file("dataset_SignLanguage.zip")

# lista de arquivos


# laço de repetição para verificar os arquivos do dataset e organiza-los
import streamlit as st
import tensorflow as tf
import os
import pandas as pd
from reconhecimento import Ligar_webcam


#from ReconhecimentoDeSinais.main import extrair_e_exibir_imagens
## from modelo_treinamento import 

#setando o layout da pagina
st.set_page_config(layout="wide")

#setup sidebar

with st.sidebar:
    st.image("https://cdn.sanity.io/images/tlr8oxjg/production/ada93729daf922ad0318c8c0295e5cb477921808-1456x816.png?w=3840&q=100&fit=clip&auto=format")
    st.header("Redes Neurais")
    st.title("Reconhecimento da Linguagem de Sinais Brasileira")
    info = """
    Este trabalho utiliza técnicas avançadas de Machine Learning para criar um sistema capaz de 
    reconhecer e interpretar a Linguagem Brasileira de Sinais (Libras). A abordagem envolve a
    coleta de dados multimodais, incluindo vídeos de sinais realizados por usuários fluentes em
    Libras. Utilizando algoritmos de visão computacional e redes neurais, o sistema aprende a
    mapear gestos específicos para suas correspondentes expressões na linguagem de sinais. Os
    resultados promissores indicam um avanço significativo na comunicação inclusiva para surdos
    e na aplicação de tecnologias de Machine Learning para problemas sociais relevantes.
    """
    
    st.info(info)
    
    #função de capturar o video
    
    st.wirite("Clique no botão para exibir a webcan")
    if st.button("Inicie a webcan"):
        Ligar_webcam()
        
        
    
    
    
    
 
    
    
        

    
    
    
    
    
   
    
    

















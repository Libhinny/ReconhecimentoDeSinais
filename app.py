import streamlit as st
import tensorflow as tf
from ReconhecimentoDeSinais.main import extrair_e_exibir_imagens
## from modelo_treinamento import 

# Cabeçalho

st.header("Redes Neurais")

# Titulo
st.title("Reconhecimento da Linguagem de Sinais Brasileira")


# Texto
texto_intro = """
A Linguagem Brasileira de Sinais (Libras) é uma língua gestual oficial no Brasil desde 2002, reconhecida pela Lei 10.436.
Diferente da língua portuguesa, a Libras tem gramática própria, utilizando gestos e expressões para formar frases.
Essa linguagem desempenha um papel crucial na inclusão social dos surdos, permitindo comunicação em diversos contextos.
"""

# Exibindo o texto inicial justificado
st.markdown(f"<div style='text-align: justify'>{texto_intro}</div>", unsafe_allow_html=True)

# Parágrafo sobre os avanços e desafios
texto_avancos_desafios = """
Apesar dos avanços, ainda há desafios para garantir uma inclusão plena, exigindo conscientização, formação de intérpretes
e práticas inclusivas. A Libras não é apenas uma ferramenta de comunicação; é um elemento vital na construção de uma
sociedade mais inclusiva e igualitária.
"""

# Exibindo o parágrafo sobre avanços e desafios justificado
st.markdown(f"<div style='text-align: justify'>{texto_avancos_desafios}</div>", unsafe_allow_html=True)

# Conclusão
texto_conclusao = """
A Libras desempenha um papel fundamental na quebra de barreiras de comunicação e na promoção da inclusão social.
Desenvolver a conscientização sobre a importância da Libras, investir na formação de intérpretes e promover práticas
inclusivas são passos essenciais para garantir que a sociedade seja acessível a todos, independentemente da capacidade auditiva.
"""

# Exibindo a conclusão justificada
st.markdown(f"<div style='text-align: justify'>{texto_conclusao}</div>", unsafe_allow_html=True)

#  Função para carregar o modelo usando o decorador @st.cache


# @st.cache(allow_output_mutation=True)
# def load_model():
  
#     model = tf.keras.models.load_model('caminho/para/seu/modelo')  # Substitua pelo caminho do seu modelo
#     return model

# # Carregando o modelo
# model = load_model()

# Agora você pode usar o modelo para realizar previsões ou outras tarefas
# Exemplo de uso:
# result = model.predict(input_data)
########################################################################################################################













import zipfile
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil

def extract_and_display_images(zip_file_path, num_images_to_display=2):
    # diretorio temporario para a extração
    temp_extracted_dir = 'temp_extracted_images'

    os.makedirs(temp_extracted_dir, exist_ok=True)

    try:
        # entraindo o arquivo zip
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # lista todos os arquivos extraídos
            extracted_files = zip_ref.namelist()

            # Exibir as primeiras N imagens do diretório
            for i, file_name in enumerate(extracted_files[:num_images_to_display]):
                file_path = os.path.join(temp_extracted_dir, file_name)

                # Extrair o arquivo para o diretório temporário
                zip_ref.extract(file_name, temp_extracted_dir)

                # Ler a imagem do arquivo extraído
                with Image.open(file_path) as img:
                    # Exibir a imagem
                    plt.subplot(1, num_images_to_display, i + 1)
                    plt.imshow(img)
                    plt.axis('off')

            # Exibir o plot
            plt.show()

    except Exception as e:
        print(f"Falha ao processar o arquivo zip: {str(e)}")

    finally:
        # Remover o diretório temporário de extração
        shutil.rmtree(temp_extracted_dir, ignore_errors=True)

zip_file_path = 'C:/Users/fllsa/Music/redes/ReconhecimentoDeSinais/archive.zip'

# Número de imagens para exibir
num_images_to_display = 3 

# Chamar a função para extrair e exibir as imagens
extract_and_display_images(zip_file_path, num_images_to_display)

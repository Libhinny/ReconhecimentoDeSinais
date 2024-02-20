import zipfile #extrair o zip
from io import BytesIO #reconhecer numeros binarios e converter 
from PIL import Image 
import matplotlib.pyplot as plt #plotar graficos 
import os #SO do computador 
import shutil #para manipulação de arquivos e diretorios 
import pandas as pd 
from sklearn.preprocessing import StandardScaler # padronizar os dados
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import random


for dirname, _, filenames in os.walk(r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# train dataset
bank_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\bank'
bus_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\bus'
car_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\car'
formation_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\formation'
hospital_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\hospital'
I_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\I'
man_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\man'
motorcycle_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\motorcycle'
my_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\my'
supermarket_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\supermarket'
we_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\we'
woman_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\woman'
you_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\you'
youPlural_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\you (plural)'
your_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\your'

bank_image = os.listdir(bank_dir)
bus_image = os.listdir(bus_dir)
car_image = os.listdir(car_dir)
formation_image = os.listdir(formation_dir)
hospital_image = os.listdir(hospital_dir)
I_image = os.listdir(I_dir)
man_image = os.listdir(man_dir)
motorcycle_image = os.listdir(motorcycle_dir)
my_image = os.listdir(my_dir)
supermarket_image = os.listdir(supermarket_dir)
we_image = os.listdir(we_dir)
woman_image = os.listdir(woman_dir)
you_image = os.listdir(you_dir)
youPlural_image = os.listdir(youPlural_dir)
your_image = os.listdir(your_dir)
motorcycle_image = os.listdir(motorcycle_dir)
youPlural_image = os.listdir(youPlural_dir)

 
print(len(bank_dir),len(bus_dir), len(car_image), len(formation_image), len(hospital_image), len(I_image), len(man_image), 
len(motorcycle_image), len(my_image), len(supermarket_image), len(we_image), len(you_image), len(youPlural_image), len(your_image))

# Making train val split
train_bank_image = bank_image[:int(.8*(len(bank_image)))]
val_bank_image = bank_image[int(.8*(len(bank_image))):]

# train_negative_images = negative_images[:int(.8*(len(negative_images)))]
# val_negative_images = negative_images[int(.8*(len(negative_images))):]

train_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\bank'
val_dir = r'C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\validation\validation'

os.makedirs(train_dir + 'bank/')
#os.makedirs(train_dir + 'Negative/')


os.makedirs(val_dir + 'Bank/')
#os.makedirs(val_dir + 'Negative/')

for image in train_bank_image:
    src = bank_dir + image
    dst = train_dir + 'bank/'
    shutil.copy(src, dst)
    
# for image in train_negative_images:
#     src = negative_dir + image
#     dst = train_dir + 'Negative/'
#     shutil.copy(src, dst)
    
for image in val_bank_image:
    src = bank_dir + image
    dst = val_dir + 'Bank/'
    shutil.copy(src, dst)
    
# for image in val_negative_images:
#     src = negative_dir + image
#     dst = val_dir + 'Negative/'
#     shutil.copy(src, dst)
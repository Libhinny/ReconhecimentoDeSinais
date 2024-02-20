import zipfile  # extrair o zip
from io import BytesIO  # reconhecer numeros binarios e converter
from PIL import Image
import matplotlib.pyplot as plt  # plotar graficos
import os  # SO do computador
import shutil  # para manipulação de arquivos e diretorios
import pandas as pd
from sklearn.preprocessing import StandardScaler  # padronizar os dados
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import random


for dirname, _, filenames in os.walk(
    r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train"
):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# train dataset
bank_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\bank"
bus_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\bus"
car_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\car"
formation_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\formation"
hospital_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\hospital"
I_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\I"
man_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\man"
motorcycle_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\motorcycle"
my_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\my"
supermarket_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\supermarket"
we_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\we"
woman_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\woman"
you_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\you"
youPlural_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\you (plural)"
your_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\your"

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


print(
    len(bank_dir),
    len(bus_dir),
    len(car_image),
    len(formation_image),
    len(hospital_image),
    len(I_image),
    len(man_image),
    len(motorcycle_image),
    len(my_image),
    len(supermarket_image),
    len(we_image),
    len(you_image),
    len(youPlural_image),
    len(your_image),
)

# Making train val split
train_man_image = man_image[: int(0.8 * len(man_image))]
val_man_image = man_image[int(0.8 * len(man_image)) :]


train_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\30 FPS\30 FPS\train\man"
val_dir = r"C:\Users\fllsa\Desktop\Redes\ReconhecimentoDeSinais\dataset_SignLanguage1\validation\validation"

# Create directories
os.makedirs(os.path.join(train_dir, "man"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "man"), exist_ok=True)
# Copy images to train directory
for image in train_man_image:
    src = os.path.join(man_dir, image)
    dst = os.path.join(train_dir, "man", image)
    shutil.copy(src, dst)

# Copy image to validation directory
for image in val_man_image:
    src = os.path.join(man_dir, image)
    dst = os.path.join(val_dir, "man", image)
    shutil.copy(src, dst)


def view_random_image(target_dir, target_class):
    target_folder = os.path.join(target_dir, target_class)
    random_image = random.choice(os.listdir(target_folder))

    img = mpimg.imread(os.path.join(target_folder, random_image))
    print(img.shape)
    plt.title(target_class)
    plt.imshow(img)
    plt.axis("off")


# View a random image
view_random_image(train_dir, "man")

train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

train_dataset = train_data_gen.flow_from_directory(
    train_dir, target_size=(227, 227), class_mode="categorical"
)

val_dataset = val_data_gen.flow_from_directory(
    val_dir, target_size=(227, 227), class_mode="categorical"
)

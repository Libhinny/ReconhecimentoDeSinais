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
    r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train"
):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# train dataset
bank_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\bank"
bus_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\bus"
car_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\car"
formation_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\formation"
hospital_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\hospital"
I_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\I"
man_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\man"
motorcycle_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\motorcycle"
my_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\my"
supermarket_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\supermarket"
we_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\we"
woman_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\woman"
you_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\you"
youPlural_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\you (plural)"
your_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train\your"

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

# Making train val split (BANK SIGN)
train_bank_image = bank_image[: int(0.8 * len(bank_image))]
val_bank_image = bank_image[int(0.8 * len(bank_image)) :]

# Making train val split (BUS SIGN)
train_bus_image = bus_image[: int(0.8 * len(bus_image))]
val_bus_image = bus_image[int(0.8 * len(bus_image)) :]

# Making train val split (CAR SIGN)
train_car_image = car_image[: int(0.8 * len(car_image))]
val_car_image = car_image[int(0.8 * len(car_image)) :]

# Making train val split (FORMATION SIGN)
train_formation_image = formation_image[: int(0.8 * len(formation_image))]
val_formation_image = formation_image[int(0.8 * len(formation_image)) :]

# Making train val split (HOSPITAL SIGN)
train_hospital_image = hospital_image[: int(0.8 * len(hospital_image))]
val_hospital_image = hospital_image[int(0.8 * len(hospital_image)) :]

# Making train val split (I SIGN)
train_I_image = I_image[: int(0.8 * len(I_image))]
val_I_image = I_image[int(0.8 * len(I_image)) :]

# Making train val split (MAN SIGN)
train_man_image = man_image[: int(0.8 * len(man_image))]
val_man_image = man_image[int(0.8 * len(man_image)) :]

# Making train val split (MOTORCYCLE SIGN)
train_motorcycle_image = motorcycle_image[: int(0.8 * len(motorcycle_image))]
val_motorcycle_image = motorcycle_image[int(0.8 * len(motorcycle_image)) :]

# Making train val split (MY SIGN)
train_my_image = my_image[: int(0.8 * len(my_image))]
val_my_image = my_image[int(0.8 * len(my_image)) :]

# Making train val split (SUPERMARKET SIGN)
train_supermarket_image = supermarket_image[: int(0.8 * len(supermarket_image))]
val_supermarket_image = supermarket_image[int(0.8 * len(supermarket_image)) :]

# Making train val split (WE SIGN)
train_we_image = we_image[: int(0.8 * len(we_image))]
val_we_image = we_image[int(0.8 * len(we_image)) :]

# Making train val split (YOU SIGN)
train_you_image = you_image[: int(0.8 * len(you_image))]
val_you_image = you_image[int(0.8 * len(you_image)) :]

# Making train val split (YOU PLURAL SIGN)
train_youPlural_image = youPlural_image[: int(0.8 * len(youPlural_image))]
val_youPlural_image = youPlural_image[int(0.8 * len(youPlural_image)) :]

# Making train val split (YOUR SIGN)
train_your_image = your_image[: int(0.8 * len(your_image))]
val_your_image = your_image[int(0.8 * len(your_image)) :]


train_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\30 FPS\30 FPS\train"
val_dir = r"C:\Users\libhi\Projeto_SignLanguage\dataset_SignLanguage\validation\validation"

# Create directories (BANK)
os.makedirs(train_dir + "\\bank", exist_ok=True)
os.makedirs(val_dir + "\\bank", exist_ok=True)



# Copy images to train directory (bank)
for image in train_bank_image:
    src = bank_dir + "\\" + image
    dst = train_dir + "\\bank"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (bank)
for image in val_bank_image:
    src = bank_dir + "\\" + image
    dst = val_dir + "\\bank"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass



# Create directories (BUS)
os.makedirs(train_dir + "\\bus", exist_ok=True)
os.makedirs(val_dir + "\\bus", exist_ok=True)

# Copy images to train directory (bus)
for image in train_bus_image:
    src = bus_dir + "\\" + image
    dst = train_dir + "\\bus"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (bus)
for image in val_bus_image:
    src = bus_dir + "\\" + image
    dst = val_dir + "\\bus"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (CAR)
os.makedirs(train_dir + "\\car", exist_ok=True)
os.makedirs(val_dir + "\\car", exist_ok=True)

# Copy images to train directory (car)
for image in train_car_image:
    src = car_dir + "\\" + image
    dst = train_dir + "\\car"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (car)
for image in val_car_image:
    src = car_dir + "\\" + image
    dst = val_dir + "\\car"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (FORMATION)
os.makedirs(train_dir + "\\formation", exist_ok=True)
os.makedirs(val_dir + "\\formation", exist_ok=True)

# Copy images to train directory (formation)
for image in train_formation_image:
    src = formation_dir + "\\" + image
    dst = train_dir + "\\formation"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (formation)
for image in val_formation_image:
    src = formation_dir + "\\" + image
    dst = val_dir + "\\formation"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (HOSPITAL)
os.makedirs(train_dir + "\\hospital", exist_ok=True)
os.makedirs(val_dir + "\\hospital", exist_ok=True)

# Copy images to train directory (hospital)
for image in train_hospital_image:
    src = hospital_dir + "\\" + image
    dst = train_dir + "\\hospital"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (hospital)
for image in val_hospital_image:
    src = hospital_dir + "\\" + image
    dst = val_dir + "\\hospital"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (I)
os.makedirs(train_dir + "\\I", exist_ok=True)
os.makedirs(val_dir + "\\I", exist_ok=True)

# Copy images to train directory (I)
for image in train_I_image:
    src = I_dir + "\\" + image
    dst = train_dir + "\\I"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (I)
for image in val_I_image:
    src = I_dir + "\\" + image
    dst = val_dir + "\\I"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (MAN)
os.makedirs(train_dir + "\man", exist_ok=True)
os.makedirs(val_dir + "\man", exist_ok=True)

# Copy images to train directory (MAN)
for image in train_man_image:
    src = man_dir + "\\" + image
    dst = train_dir + "\man"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (MAN)
for image in val_man_image:
    src = man_dir + "\\" + image
    dst = val_dir + "\man"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (MOTORCYCLE)
os.makedirs(train_dir + "\motorcycle", exist_ok=True)
os.makedirs(val_dir + "\motorcycle", exist_ok=True)

# Copy images to train directory (motorcycle)
for image in train_motorcycle_image:
    src = motorcycle_dir + "\\" + image
    dst = train_dir + "\motorcycle"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (motorcycle)
for image in val_motorcycle_image:
    src = motorcycle_dir + "\\" + image
    dst = val_dir + "\motorcycle"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Create directories (MY)
os.makedirs(train_dir + "\my", exist_ok=True)
os.makedirs(val_dir + "\my", exist_ok=True)

# Copy images to train directory (my)
for image in train_my_image:
    src = my_dir + "\\" + image
    dst = train_dir + "\my"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (my)
for image in val_my_image:
    src = my_dir + "\\" + image
    dst = val_dir + "\my"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (SUPERMARKET)
os.makedirs(train_dir + "\supermarket", exist_ok=True)
os.makedirs(val_dir + "\supermarket", exist_ok=True)

# Copy images to train directory (supermarket)
for image in train_supermarket_image:
    src = supermarket_dir + "\\" + image
    dst = train_dir + "\supermarket"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (supermarket)
for image in val_supermarket_image:
    src = supermarket_dir + "\\" + image
    dst = val_dir + "\supermarket"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (we)
os.makedirs(train_dir + "\we", exist_ok=True)
os.makedirs(val_dir + "\we", exist_ok=True)

# Copy images to train directory (we)
for image in train_we_image:
    src = we_dir + "\\" + image
    dst = train_dir + "\we"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (we)
for image in val_we_image:
    src = we_dir + "\\" + image
    dst = val_dir + "\we"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (you)
os.makedirs(train_dir + "\you", exist_ok=True)
os.makedirs(val_dir + "\you", exist_ok=True)

# Copy images to train directory (you)
for image in train_you_image:
    src = you_dir + "\\" + image
    dst = train_dir + "\you"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (you)
for image in val_you_image:
    src = you_dir + "\\" + image
    dst = val_dir + "\you"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (youPlural)
os.makedirs(train_dir + "\youPlural", exist_ok=True)
os.makedirs(val_dir + "\youPlural", exist_ok=True)

# Copy images to train directory (youPlural)
for image in train_youPlural_image:
    src = youPlural_dir + "\\" + image
    dst = train_dir + "\youPlural"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (youPlural)
for image in val_youPlural_image:
    src = youPlural_dir + "\\" + image
    dst = val_dir + "\youPlural"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


# Create directories (your)
os.makedirs(train_dir + "\your", exist_ok=True)
os.makedirs(val_dir + "\your", exist_ok=True)

# Copy images to train directory (your)
for image in train_your_image:
    src = your_dir + "\\" + image
    dst = train_dir + "\your"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass

# Copy image to validation directory (your)
for image in val_your_image:
    src = your_dir + "\\" + image
    dst = val_dir + "\your"
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass



# def view_random_image(target_dir, target_class):
#     target_folder = os.path.join(target_dir, target_class)
#     random_image = random.choice(os.listdir(target_folder))

#     img = mpimg.imread(os.path.join(target_folder, random_image))
#     print(img.shape)
#     plt.title(target_class)
#     plt.imshow(img)
#     plt.axis("off")


# # View a random image
# view_random_image(train_dir, 'man')

# train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
# val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)

# train_dataset = train_data_gen.flow_from_directory(
#     train_dir, target_size=(227, 227), class_mode="categorical"
# )

# val_dataset = val_data_gen.flow_from_directory(
#     val_dir, target_size=(227, 227), class_mode="categorical"
# )

import os
import shutil

# Diretório de origem e destino para treinamento
you_train_dir = r"C:\Users\ytalo\Downloads\dataset_Libras\30 FPS\30 FPS\train\you"
new_you_train_dir = r"C:\Users\ytalo\Downloads\dataset_Libras\30 FPS\30 FPS\train\you_renamed"
os.makedirs(new_you_train_dir, exist_ok=True)

# Loop through images in the source directory
for filename in os.listdir(you_train_dir):
    # Source path for the image
    src = os.path.join(you_train_dir, filename)

    # Check if the file is a file (not a directory)
    if os.path.isfile(src):
        # Extract filename and extension
        base_name, extension = os.path.splitext(filename)

        # Check if it's a valid image extension
        if extension.lower() not in [".jpg", ".jpeg", ".png"]:
            # Add the ".jpeg" extension to the image name
            new_name = base_name + ".jpeg"

            # Ensure the new name is unique in the destination directory
            count = 1
            while os.path.exists(os.path.join(new_you_train_dir, new_name)):
                new_name = f"{base_name}_{count}.jpeg"
                count += 1

            # Destination path for the image with the new name and extension
            dst = os.path.join(new_you_train_dir, new_name)

            # Copy the image with the new name and extension to the destination directory
            try:
                shutil.copy2(src, dst)
            except (OSError, shutil.Error) as e:
                print(f"Error occurred while copying file: {e}")

# Exibir as imagens renomeadas na nova pasta de treinamento
renamed_train_files = os.listdir(new_you_train_dir)
print("Imagens renomeadas no diretório de treinamento:")
for filename in renamed_train_files:
    print(filename)

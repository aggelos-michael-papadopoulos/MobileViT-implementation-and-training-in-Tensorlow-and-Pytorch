import pandas as pd
import os
import shutil

# we first created two temp folders (temp_train, temp_val) to store all the images, by the condition
# if they are for train or val according to the caltech_data.csv file.
# And then we create the train and
# val folders and pass the images there (according to each specific class)
data = pd.read_csv('/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/caltech_data.csv')

temp_train_folder = '/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/temp_train/'
temp_val_folder = '/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/temp_val/'

train_folder = '/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/train/'
val_folder = '/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/val/'

completed = False
# run this once to place the classes into train and val folder
if completed:
    for i in os.listdir('/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/256_ObjectCategories/'):
        os.mkdir(train_folder + i)
        os.mkdir(val_folder + i)

# Run this once to seperate to train and val folders from all the images in the csv file
if completed:
    for i in range(len(data)):
        img = data['full_paths'][i]
        if data['execution'][i] == 'train':
            shutil.move(img, train_folder)

        elif data["execution"][i] == "eval":
            shutil.move(img, val_folder)

class_list = [class_name for class_name in os.listdir(train_folder)]
class_list.sort()


# separate everything to each corresponding folder
# we move all the images the temp_train folder to the train_folder and its corresponding class
if completed:
    for cls in class_list:
        for img in os.listdir(temp_train_folder):
            print(img)
            if img.split('_')[0] == cls.split('.')[0]:
                shutil.move((temp_train_folder + img), (train_folder + cls))

# separate everything to each corresponding folder
# we move all the images the temp_val folder to the val_folder and its corresponding class
if not completed:
    for cls in class_list:
        for img in os.listdir(temp_val_folder):
            print(img)
            if img.split('_')[0] == cls.split('.')[0]:
                shutil.move((temp_val_folder + img), (val_folder + cls))
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder

def check_duplicates( give_me_a_list):
    import collections
    hope = [item for item, count in collections.Counter(give_me_a_list).items() if count > 1]
    if hope:
        print('Oh shoot, there is a duplicate!')
    else:
        print('Sector Clear!')

# Here put the path on which caltech_256 has been downloaded       
caltech_path = '/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/256_ObjectCategories'

classes_paths = sorted(glob.glob(caltech_path+'/*'))
classes = [i.split('/')[-1] for i in classes_paths]

encoder = LabelEncoder()
labels = encoder.fit_transform(classes)

class_label = dict(zip(classes, labels))

full_paths = []
list_classes = []
labels = []
execution = []
for path in classes_paths:
    temp_class = path.split('/')[-1]
    images = glob.glob(path+'/*.jpg')
    max_images = len(images)
    train_index = int(0.8*max_images)
    for i in range(train_index):
        full_paths.append(images[i])
        list_classes.append(temp_class)
        labels.append(class_label[temp_class])
        execution.append('train')
    for i in range(train_index, max_images):
        full_paths.append(images[i])
        list_classes.append(temp_class)
        labels.append(class_label[temp_class])
        execution.append('eval')

check_duplicates(full_paths)

df = pd.DataFrame({'full_paths': full_paths,
                   'classes': list_classes,
                   'labels': labels,
                   'execution': execution})
print(df.head(25))

# where do you want to save your csv file
df.to_csv('/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/caltech_data.csv')



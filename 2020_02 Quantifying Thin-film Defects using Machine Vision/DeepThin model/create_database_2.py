import os
from PIL import Image
import pickle
import numpy as np

dataset_positive = 'C:\\Deep Learn\\DataSet2\\Positive\\'
dataset_negative = 'C:\\Deep Learn\\DataSet2\\Negative\\'

image_height = 256
images = list()
classes = list()
meta = list()

for i in os.listdir(dataset_positive):
    if i.endswith('jpg'):
        jpgfile = Image.open(dataset_positive + i).convert('L')
        meta.append(i)
        aspect = jpgfile.size[0] / jpgfile.size[1]
        out = jpgfile.resize((int(aspect * image_height), image_height), Image.ANTIALIAS)
        pixels = np.asarray(out)
        images.append(pixels)
        classes.append(1)
        rotated_image = out.rotate(180)
        pixels = np.asarray(rotated_image)
        images.append(pixels)
        classes.append(1)
        meta.append(i)
        rotated_image = out.transpose(Image.FLIP_LEFT_RIGHT)
        pixels = np.asarray(rotated_image)
        images.append(pixels)
        classes.append(1)
        meta.append(i)
        rotated_image = rotated_image.rotate(180)
        pixels = np.asarray(rotated_image)
        images.append(pixels)
        classes.append(1)
        meta.append(i)

for i in os.listdir(dataset_negative):
    if i.endswith('jpg'):
        jpgfile = Image.open(dataset_negative + i).convert('L')
        meta.append(i)
        aspect = jpgfile.size[0] / jpgfile.size[1]
        out = jpgfile.resize((int(aspect * image_height), image_height), Image.ANTIALIAS)
        pixels = np.asarray(out)
        images.append(pixels)
        classes.append(0)

X_Train = np.asarray(images)
Y_Train = np.asarray(classes)

# out.show()


with open('X_Train2.data', 'wb') as filehandle:
    pickle.dump(X_Train, filehandle)
with open('Y_Train2.data', 'wb') as filehandle:
    pickle.dump(Y_Train, filehandle)
with open('Meta2.data', 'wb') as filehandle:
    pickle.dump(meta, filehandle)

# with open('listfile.data', 'rb') as filehandle:
    # read the data as binary data stream
#    placesList = pickle.load(filehandle)

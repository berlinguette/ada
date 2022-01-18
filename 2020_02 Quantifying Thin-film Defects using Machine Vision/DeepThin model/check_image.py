import os
from PIL import Image
import pickle

dataset_positive = 'C:\\Deep Learn\\DataSet\\Positive\\'
dataset_negative = 'C:\\Deep Learn\\DataSet\\Negative\\'

image_height = 512
images = list()
for i in os.listdir(dataset_negative):
    if i.endswith('jpg'):
        images.append(i)
jpgfile = Image. open(dataset_negative+images[0]).convert('LA')
aspect = jpgfile.size[0]/jpgfile.size[1]
out = jpgfile.resize((int(aspect*image_height), image_height), Image.ANTIALIAS)
out.show()

pixels = list(out.getdata())

with open('listfile.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(pixels, filehandle)

with open('listfile.data', 'rb') as filehandle:
    # read the data as binary data stream
    placesList = pickle.load(filehandle)

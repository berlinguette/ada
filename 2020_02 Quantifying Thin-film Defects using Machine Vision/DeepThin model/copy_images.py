import csv
import os
from shutil import copyfile

images_directory = 'C:\\Deep Learn\\Images\\'
dataset_positive = 'C:\\Deep Learn\\DataSet\\Positive\\'
dataset_negative = 'C:\\Deep Learn\\DataSet\\Negative\\'

with open('C:\\Deep Learn\\Data\\FreserResult.csv', 'r') as csvFile:
    reader = list(csv.reader(csvFile))
    no_file_count = 0;
    for row in reader:
        if row[2].isdigit():
            # if int(row[2]) == 0 and int(row[3]) == 0 and int(row[4]) == 0 and int(row[5]) == 0:
            if int(row[3]) == 0:    # No Crack
                if os.path.isfile(images_directory + row[0]):
                    print(row[0])
                    copyfile(images_directory + row[0], dataset_positive+row[0])
                else:
                    no_file_count += 1
            else:
                if os.path.isfile(images_directory + row[0]):
                    copyfile(images_directory + row[0], dataset_negative + row[0])
        else:
            print('Not Number: '+row[2])

    print('Number of No File: '+str(no_file_count))

    #print(reader)


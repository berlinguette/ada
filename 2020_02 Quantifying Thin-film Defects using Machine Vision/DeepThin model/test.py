import csv
import os
data = []

with open('C:\Deep Learn\Data\Kevanresult.csv', 'r') as csvFile:
    reader =list(csv.reader(csvFile))
    count = 0;
    file_count = 0;
    for row in reader:
        if(row[2]=='0'  and row[3]=='0' and row[4]=='0' and row[5]=='0'):
            #print(row[0])
            count+=1
        if (not os.path.isfile('C:\Deep Learn\Images\\' + row[0])):
            print(row[0])
            file_count+=1

    print(count)
    print(file_count)
csvFile.close()

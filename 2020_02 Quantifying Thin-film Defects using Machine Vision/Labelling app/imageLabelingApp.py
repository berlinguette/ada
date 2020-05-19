from labelingGui import Labeler
import os


images_directory = 'D:\\Deep Learn\\Images'
result_file_name = 'result.csv'


def main():
    images = list()
    for i in os.listdir(images_directory):
        if i.endswith('jpg'):
            images.append(i)
    os.chdir(images_directory)
    l = Labeler(images, result_file_name)
    l.show_panel()


main()

from tkinter import *
from PIL import ImageTk, Image
import csv
import os


class Labeler:
    global img_canv
    global render
    canv_width = 800
    canv_height = 600

    def __init__(self, images, result_file_name):
        self.result_file_name = result_file_name
        self.image_list = images
        self.image_index = 0
        self.root = Tk()
        self.frame = Frame(self.root)
        self.root.title('Image labeling app')
        self.previous = Button(self.frame, text='Previous', command=self.previous_call_back)
        self.next = Button(self.frame, text='Next', command=self.next_call_back)
        self.crack_scale = Scale(self.frame, variable=int, orient=HORIZONTAL, to=10,
                                 label='Crack',
                                 length=200)
        self.dust_scale = Scale(self.frame, variable=int, orient=HORIZONTAL, to=10,
                                label='Dust',
                                length=200)
        self.dewetting_scale = Scale(self.frame, variable=int, orient=HORIZONTAL, to=10,
                                     label='Dewetting/film not covering', length=200)
        self.cloudiness_scale = Scale(self.frame, variable=int, orient=HORIZONTAL, to=10,
                                      label='Cloudiness/Smokiness', length=200)
        self.note = Entry(self.frame, justify=CENTER, width=100)
        self.note_label = Label(self.frame, text='Note')
        self.canv = Canvas(self.frame, relief=RIDGE)

    def apply_changes(self):
        header = ['file', 'note', 'dust', 'crack', 'dewetting', 'cloudiness']
        if not os.path.isfile(self.result_file_name):
            self.write_in_file(header)

        file_name = self.image_list[self.image_index]
        row = [file_name, self.note.get(), self.dust_scale.get(), self.crack_scale.get(), self.dewetting_scale.get(),
               self.cloudiness_scale.get()]
        self.write_in_file(row)

    def write_in_file(self, row):
        with open(self.result_file_name, 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)

    def p_switch(self):
        if self.image_index == 0:
            self.previous.config(state='disabled')
        else:
            self.previous.config(state='normal')

    def n_switch(self):
        if self.image_index == len(self.image_list) - 1:
            self.next.config(state='disabled')
        else:
            self.next.config(state='normal')

    def previous_call_back(self):
        self.apply_changes()
        self.reset_values()
        self.image_index -= 1
        self.p_switch()
        self.n_switch()
        self.change_image()

    def change_image(self):
        image = Image.open(self.image_list[self.image_index])
        self.render = ImageTk.PhotoImage(self.resize_image(image))
        self.canv.itemconfigure(self.img_canv, image=self.render)

    def next_call_back(self):
        self.apply_changes()
        self.reset_values()
        self.image_index += 1
        self.n_switch()
        self.p_switch()
        self.change_image()

    def reset_values(self):
        self.note.delete(0, 'end')
        self.crack_scale.set(0)
        self.dust_scale.set(0)
        self.cloudiness_scale.set(0)
        self.dewetting_scale.set(0)

    def show_panel(self):
        self.frame.grid(row=0, column=0)
        image = Image.open(self.image_list[self.image_index])
        self.render = ImageTk.PhotoImage(self.resize_image(image))
        self.canv.config(width=670, height=500)
        self.img_canv = self.canv.create_image(0, 0, anchor=NW, image=self.render)
        self.canv.grid(row=0, column=1, columnspan=5)
        self.dust_scale.grid(row=1, column=0)
        self.crack_scale.grid(row=1, column=1)
        self.cloudiness_scale.grid(row=1, column=4)
        self.dewetting_scale.grid(row=1, column=6)
        self.note_label.grid(row=2, column=0)
        self.note.grid(row=2, column=1, columnspan=6)
        self.previous.grid(row=3, column=0)
        self.next.grid(row=3, column=6)

        self.p_switch()
        self.n_switch()
        self.root.mainloop()

    def resize_image(self, img):
        return img.resize((self.canv_width, self.canv_height), Image.ANTIALIAS)

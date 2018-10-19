from matplotlib.image import imsave, imread
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np

from image import Image as IMG
from PIL import ImageTk, Image


class Window(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self._buttons = dict()
        self._clicks = dict()
        self._entries = dict()
        self._labels = dict()
        self._master = master
        self._img = IMG(np.zeros((1, 1))+255)
        self._init_window()

    def _load_image(self):
        path = self._entries['path'].get()
        pic = imread("img/"+path)
        self._img = IMG(pic)
        self._img.to_binary()
        self._show_image()

    def _show_image(self):
        render = ImageTk.PhotoImage(Image.fromarray(self._img.get_matrix()))
        if 'image' not in self._entries.keys():
            temp_img = tk.Label(self, image=render)
        else:
            temp_img = self._entries['image']
        temp_img.configure(image=render)
        temp_img.image = render
        temp_img.pack()
        self._entries['image'] = temp_img

    def _erosion(self):
        self._img.bin_erosion()
        self._show_image()

    def _dilation(self):
        self._img.bin_dilation()
        self._show_image()

    def _init_window(self):

        self.pack(fill=tk.BOTH, expand=1)

        path = tk.StringVar()
        path.set('face.jpg')
        self._entries['path'] = tk.Entry(self, textvar=path)

        self._labels['path'] = tk.Label(self, text='Path:')

        self._buttons["load_img"] = tk.Button(self, text='Load Image', width=10, height=1,
                                              command=self._load_image)
        self._buttons['erosion'] = tk.Button(self, text='Erosion', width=10, height=1,
                                             command=self._erosion)
        self._buttons['dilation'] = tk.Button(self, text='Dilation', width=10, height=1,
                                             command=self._dilation)
        self._show_image()

        for item in self._labels.values():
            item.pack()
        for item in self._entries.values():
            item.pack()
        for item in self._buttons.values():
            item.pack()


from matplotlib.image import imsave, imread
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
import threading

from image import Image as IMG
from PIL import ImageTk, Image


class Window(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self._buttons = dict()
        self._texts = dict()
        self._clicks = dict()
        self._entries = dict()
        self._labels = dict()
        self._master = master
        self._img = IMG(np.zeros((1, 1))+255)
        self._init_window()

    def string2array(self, raw):
        try:
            raw = raw.strip()
            raw = raw.replace(' ', '')
            lines = raw.split('\n')
            height = len(lines)
            width = len(lines[0].split(','))
            result = np.zeros((height, width))
            for i in range(height):
                ele = lines[i].split(',')
                for j in range(width):
                    result[i, j] = ele[j]
            return result
        except:
            print("Invalid Kernel")
            return -1

    def _load_image_base(self, path):
        pic = imread("img/" + path)
        self._img = IMG(pic)
        self._img.to_binary()
        self._show_image()

    def _load_image(self):
        # start a new thread to avoid stuck in the main window
        path = self._entries['path'].get()
        t1 = threading.Thread(target=self._load_image_base, args=(path,))
        t1.start()

    def _show_image(self):
        wid = int(500 * self._img.get_matrix().shape[1] / self._img.get_matrix().shape[0])
        render = ImageTk.PhotoImage(Image.fromarray(self._img.get_matrix()).resize((wid, 500)))
        if 'image' not in self._entries.keys():
            temp_img = tk.Label(self, image=render)
        else:
            temp_img = self._entries['image']

        temp_img.configure(image=render, height=500, width=wid)
        temp_img.image = render
        temp_img.pack()
        self._entries['image'] = temp_img
        plt.imsave('new_img.jpg', self._img.get_matrix(), cmap='Greys_r')

    def _erosion(self):
        kernel_raw = self._texts['kernel'].get(0.0, tk.END)
        kernel = self.string2array(kernel_raw)
        if kernel is -1:
            return
        self._img.bin_erosion(kernel=kernel)
        self._show_image()

    def _dilation(self):
        kernel_raw = self._texts['kernel'].get(0.0, tk.END)
        kernel = self.string2array(kernel_raw)
        if kernel is -1:
            return
        self._img.bin_dilation(kernel=kernel)
        self._show_image()

    def _init_window(self):

        self.pack(fill=tk.BOTH, expand=1)

        path = tk.StringVar()
        path.set('face.jpg')
        self._entries['path'] = tk.Entry(self, textvar=path)

        self._texts['kernel'] = tk.Text(self, height=5)
        self._texts['kernel'].insert(tk.END, "255, 255\n255, 255\n255, 255")

        self._labels['path'] = tk.Label(self, text='Path:')
        self._labels['path'].pack()
        self._entries['path'].pack()

        self._labels['kernel'] = tk.Label(self, text='Kernel:')
        self._labels['kernel'].pack()
        self._texts['kernel'].pack()

        self._buttons["load_img"] = tk.Button(self, text='Load Image', width=10, height=1,
                                              command=self._load_image)
        self._buttons["load_img"].pack(padx=6)
        self._buttons['erosion'] = tk.Button(self, text='Erosion', width=10, height=1,
                                             command=self._erosion)
        self._buttons['erosion'].pack(padx=6)
        self._buttons['dilation'] = tk.Button(self, text='Dilation', width=10, height=1,
                                             command=self._dilation)
        self._buttons['dilation'].pack(padx=6)
        self._show_image()


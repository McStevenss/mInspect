from tkinter import *
from tkinter import ttk, filedialog
from ttkthemes import ThemedTk
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk  # Import Image and ImageTk from Pillow
import numpy as np
from scrollable_image import ScrollableImage
import os
import io

class mInspect(Tk):
    def __init__(self):
        super().__init__()  # initialize Tk

        self.window_title = "mInspect"
        self.dimension = "1280x720"
        self.workspace = ""

        #Tabs
        self.tabs = {}
        self.tab_control = ttk.Notebook(self)
        self.loadModelButton = None
        self.extract_layer_button = None


        #Style
        self.style = ttk.Style()
        self.style.theme_use('default')

        self.menu_bar = Menu(self)

        #Model
        self.model = None
        self.isModelLoaded = False
        self.layers = []
        self.processed_filters = []

        #Images
        self.base_filter_images = {}

        #Program
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        #Window setup
        self.title(self.window_title)
        self.geometry(self.dimension)

        #File menu
        filemenu = Menu(self.menu_bar, tearoff=0)
        filemenu.add_command(label="New Workspace", command=self.openWorkSpaceFileDialog)
        self.menu_bar.add_cascade(label="File", menu=filemenu)

        #Add tabs
        self.add_tab("Main")
        self.loadModelButton = Button(self.tabs['Main'],text="Load Model", command=self.load_model)
        self.loadModelButton.grid(column=0, row=0, sticky='w')  # Align to the left

        self.extract_layer_button = Button(self.tabs['Main'],text="Extract Base-layer",state='disabled',command=self.extract_base_layer)
        self.extract_layer_button.grid(column=1, row=0, sticky='w')  # Align to the left
        self.config(menu=self.menu_bar)

        self.extract_feature_button = Button(self.tabs['Main'],text="Extract Feature-map",state='disabled',command=self.extract_feature_map)
        self.extract_feature_button.grid(column=2, row=0, sticky='w')  # Align to the left
        self.config(menu=self.menu_bar)

    def on_closing(self):
        print("Goodbye!")
        self.quit()
        self.destroy()

    def openWorkSpaceFileDialog(self):
        dir = filedialog.askdirectory()
        if dir != "":
            print(f"Workspace: {dir}")
            self.title(f"{self.title} - {dir}")
            self.workspace = dir
        else:
            print("No workspace chosen")

    def add_tab(self, tab_name):
        self.tabs[tab_name] = ttk.Frame(self.tab_control)

        #Load Tabs
        for tab_name, tab in self.tabs.items():
            self.tab_control.add(tab, text=tab_name)

        self.tab_control.pack(expand=1, fill='both')

    def load_model(self):
        self.model = VGG16()
        print("Model loaded!")
        self.isModelLoaded = True
 
        self.loadModelButton.config(state='disabled')
        self.loadModelButton.config(text="Model Loaded")
        self.extract_layer_button.config(state='normal')

        self.extract_feature_button.config(state='normal')

    def extract_base_layer(self):
        # summarize filter shapes
        print("--------Relevant Model layers--------")

        for layer in self.model.layers:
            # check for convolutional layer
            if 'conv' not in layer.name:
                continue

            # get filter weights
            filters, biases = layer.get_weights()
            print(layer.name, filters.shape)

            #save processed filters
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)

            self.layers.append([layer,filters,biases])
            #self.processed_filters.append(filters)

        print("-------------------------------------")

        layer,filter,bias = self.layers[0]
        n_filters = filter.shape[3]
        print("n_filters", n_filters)
        print("Extracting baselayer")
        fig, axs = plt.subplots(n_filters, 3, figsize=(10, 10))  # Create the combined figure
        tmp_image = np.zeros((filter.shape[0], filter.shape[1], 3), dtype=np.float32)
        #Skip input layer
        ix = 1
        for i in range(n_filters):
            # get the filter
            f = filter[:, :, :, i]
            tmp_image.fill(0)  # Reset the temporary image to black for each filter
            # plot each channel separately
            for j in range(3):

                tmp_image[:, :, j] = f[:, :, j]
                # specify subplot and turn of axis
                ax = axs[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                ax.imshow(tmp_image, cmap='gray')
                ix = ix + 1

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        #Process image
        self.add_tab("BaseLayerFilters")
        fig.savefig(f"{layer.name}.png")
        plt.close(fig)

        self.base_filter_image = Image.open(f"{layer.name}.png")
        self.base_filter_image = ImageTk.PhotoImage(self.base_filter_image)

        canvas = Canvas(self.tabs['BaseLayerFilters'])
        canvas.pack(side=LEFT,fill=BOTH,expand=1)

        image_label = Label(self.tabs['BaseLayerFilters'], image=self.base_filter_image)
        image_label.pack()#grid(row=0, column=1)  # Use grid instead of pack
        
        scrollbar = ttk.Scrollbar(self.tabs['BaseLayerFilters'], orient=VERTICAL, command=canvas.yview)
        scrollbar.pack(side=RIGHT,fill=Y)#grid(row=0, column=0, sticky='ns')  # Use sticky to make the scrollbar fill vertically

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.create_window((0, 0), window=image_label, anchor=NW)
        canvas.update_idletasks()  # Update the canvas

        canvas.config(scrollregion=canvas.bbox(ALL))

        return None
        
    def extract_feature_map(self):

        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            # check for convolutional layer
            if 'conv' not in layer.name:
                continue
            # summarize output shape
            print(i, layer.name, layer.output.shape)

if __name__ == "__main__":
    minspect = mInspect()
    minspect.mainloop()
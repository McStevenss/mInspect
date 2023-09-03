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

from keras.applications.vgg16 import preprocess_input
#from keras.preprocessing.image import load_img
from keras.utils import load_img
from keras.utils import img_to_array
#from keras.preprocessing.image import img_to_array
from keras.models import Model


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
        self.tab_images = []

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
        self.file = None
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

    def openFileDialog(self):
        filetypes = (
        ("JPEG", "*.jpg"),
        ("PNG", "*.png")
        )

        file = filedialog.askopenfile(filetypes=filetypes)
        if file != "":
            print(f"File chosen: {file}")
            return file.name
        else:
            print("No file chosen")
            return None

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

    def show_img_in_tab(self, img_path, tab_name):

        self.add_tab(f"{tab_name}")
        
        #self.base_filter_image = Image.open(img_path)
        tab_img = Image.open(img_path)
        self.tab_images.append(ImageTk.PhotoImage(tab_img))

        canvas = Canvas(self.tabs[tab_name])
        canvas.pack(side=LEFT,fill=BOTH,expand=1)

        image_label = Label(self.tabs[tab_name], image=self.tab_images[-1])
        image_label.pack()#grid(row=0, column=1)  # Use grid instead of pack
        
        scrollbar = ttk.Scrollbar(self.tabs[tab_name], orient=VERTICAL, command=canvas.yview)
        scrollbar.pack(side=RIGHT,fill=Y)#grid(row=0, column=0, sticky='ns')  # Use sticky to make the scrollbar fill vertically

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.create_window((0, 0), window=image_label, anchor=NW)
        canvas.update_idletasks()  # Update the canvas

        canvas.config(scrollregion=canvas.bbox(ALL))

        return

    def extract_base_layer(self):
        # summarize filter shapes
        print("--------Relevant Model layers--------")

        for idx, layer in enumerate(self.model.layers):
            # check for convolutional layer
            if 'conv' not in layer.name:
                continue

            # get filter weights
            filters, biases = layer.get_weights()
            print(idx, layer.name, filters.shape)

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
        fig.savefig(f"{layer.name}.png")
        plt.close(fig)

        self.show_img_in_tab(f"{layer.name}.png","BaseLayerFilters")
       
        return None
        
    def extract_feature_map(self):

        #Get image file:
        file = self.openFileDialog()
        if file == None:
            print("Need a image to extract")
            return

        ixs = [2, 5, 9, 13, 17]
        outputs = [self.model.layers[i].output for i in ixs]
        model = Model(inputs=self.model.inputs, outputs=outputs)
        img = load_img(file, target_size=(224, 224))

        # convert the image to an array
        img = img_to_array(img)
        # expand dimensions so that it represents a single 'sample'
        img = np.expand_dims(img, axis=0)

        # prepare the image (e.g. scale pixel values for the vgg)
        img = preprocess_input(img)

        # get feature map for first hidden layer
        feature_maps = model.predict(img)

        feature_map_files = []
        square = 8
        for idx, fmap in enumerate(feature_maps):
            # plot all 64 maps in an 8x8 squares
            ix = 1
            for _ in range(square):
                for _ in range(square):
                    # specify subplot and turn of axis
                    ax = plt.subplot(square, square, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
                    ix += 1
            feature_map_files.append(f"fmap-{idx}.png")
            plt.savefig(f"fmap-{idx}.png")
            plt.close()
        # show the figure
        for fmap in feature_map_files:
            self.show_img_in_tab(fmap,fmap)

if __name__ == "__main__":
    minspect = mInspect()
    minspect.mainloop()
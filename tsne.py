import numpy as np
import os
from sklearn.datasets import load_sample_images
from sklearn.manifold import TSNE
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from keras.models import Model
from keras.utils import load_img, img_to_array
from tqdm import tqdm
from tensorflow import keras 
import cv2
from sklearn.decomposition import PCA

print("Loading model...")
model = VGG16()
print("Model loaded!")

conv_layers = []

print("Processing layers")
for idx, layer in enumerate(model.layers):
    if "conv" not in layer.name:
        continue
    #print(f"{idx} | {layer.name}")    
    conv_layers.append(idx)


print("Conv layer indexes:",conv_layers)
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]

print("Creating injection model")
model = Model(inputs=model.inputs, outputs=outputs)
batch_size = 32
print("Loading dataset")
image_data = keras.utils.image_dataset_from_directory(
    directory='dataset\\raw-img',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(224, 224))

class_names = image_data.class_names
print("Classes found:",class_names)


test_data = image_data.take(3)

#class_index = np.argmax(label)
#class_name = class_names[class_index]

collected_features = []

for images, labels in tqdm(test_data,desc="Predicting on images"):

    # Initialize lists to store batch features and batch labels
    batch_features = []
    batch_labels = []

    for image, label in zip(images, labels):
        img = np.copy(image)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Append the preprocessed image and label to the batch lists
        batch_features.append(img)
        batch_labels.append(label)

        if len(batch_features) == batch_size:
                    batch_features = np.vstack(batch_features)  # Stack the batch features
                    batch_predictions = model.predict(batch_features)  # Predict on the batch
                    collected_features.extend(zip(batch_predictions, batch_labels))
                    batch_features = []
                    batch_labels = []

    if batch_features:
        batch_features = np.vstack(batch_features)
        batch_predictions = model.predict(batch_features)
        collected_features.extend(zip(batch_predictions, batch_labels))
        feature_maps = model.predict(img)


print("Collected features",len(collected_features))

test_features = []
test_labels = []
for predictions, labels in collected_features:
    #print(predictions.shape)
    for prediction in predictions:
        #print(prediction.shape)
        if prediction.shape == (14, 14, 512):
            # test_features.append(prediction)
            test_features.append(prediction)
            test_labels.append(labels)

# feature_maps_array = np.array(test_features)
#Take one sample
#feature_maps_array = np.array(test_features[:50])
feature_maps_array = np.asarray([feature for feature in test_features[:50]])

#FLATTENING
num_samples, height, width, channels = feature_maps_array.shape
flattened_feature_vectors = feature_maps_array.reshape(num_samples, -1)
print("Len   flatten", len(flattened_feature_vectors))
print("Shape flatten", flattened_feature_vectors.shape)

flattened_feature_vectors = flattened_feature_vectors[:100]

# print("Flattened feature shape",flattened_feature_vectors.shape)
# print("Flattened feature len",len(flattened_feature_vectors))

n_components = 3
perplexity = 30
tsne = TSNE(n_components=n_components,perplexity=perplexity)

# # Fit PCA to your feature vectors
tsne_result = tsne.fit_transform(flattened_feature_vectors)

fig = plt.figure()
ax = plt.axes(projection='3d')
plot_Animal = ax.scatter(tsne_result[:,0], tsne_result[:,1], tsne_result[:,2] ,color='green')
ax.set_title("T-SNE Animal")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

# displaying the plot
plt.show()

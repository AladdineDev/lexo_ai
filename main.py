import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import os

from sklearn.neural_network import MLPRegressor, MLPClassifier

IMAGE_REDUCED_SIZE = 64
FOLDERS = ['animals', 'numbers', 'flags', 'letters']

def showImage(image):
    print(image.shape)
    fig = plt.figure()
    plt.imshow(image)
    plt.show()

def loadImages(filenames):
    images = []
    for filename in filenames:
        image = skimage.io.imread(filename)
        image = skimage.transform.resize(image, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE))
        images.append(image)
    return np.array(images)

def loadDataset(folders):
    filenames = []
    classes = []
    for folder in folders:
        for filename in os.listdir('images/' + folder):
            if filename.endswith('.png'):
                filenames.append('images/' + folder + '/' + filename)
                classes.append(folder)
    return filenames, classes

if __name__ == "__main__":
    filenames, classes = loadDataset(FOLDERS)
    print(filenames)
    print(classes)
    images = loadImages(filenames)
    print(images.shape)
    showImage(images[0])
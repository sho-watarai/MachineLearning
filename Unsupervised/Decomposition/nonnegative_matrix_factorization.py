import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import NMF

np.random.seed(0)

n_dim = 10
img_height = 64
img_width = 64


if __name__ == "__main__":
    file_list = glob.glob("../../../GAN/DCGAN/faces/*.jpg")

    faces = np.zeros((len(file_list), img_height, img_width), dtype="float32")
    for i, file in enumerate(file_list):
        faces[i] = cv2.resize(cv2.imread(file, 0), (img_width, img_height))

    #
    # non-negative matrix factorization
    #
    nmf = NMF(n_components=n_dim, init="random", max_iter=10000)

    nmf.fit(faces.reshape(len(faces), -1))

    data = nmf.components_.reshape(n_dim, img_height, img_width)

    #
    # visualization
    #
    plt.figure(figsize=(12, 6))
    for i in range(n_dim):
        plt.subplot(2, 5, i + 1)
        plt.axis("off")
        plt.imshow(data[i], cmap="gray")
        plt.title("%d component" % (i + 1))
    plt.show()
    

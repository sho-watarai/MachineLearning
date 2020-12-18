import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import NMF

np.random.seed(0)

n_dim = 10
img_height = 64
img_width = 64


def nmf_face():
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


def nmf_reconstruction_error():
    file_list = glob.glob("../../../GAN/DCGAN/faces/*.jpg")

    faces = np.zeros((len(file_list), img_height, img_width), dtype="float32")
    for i, file in enumerate(file_list):
        faces[i] = cv2.resize(cv2.imread(file, 0), (img_width, img_height))

    #
    # reconstruction error
    #
    errors = []
    for i in range(1, 51):
        nmf = NMF(n_components=i, init="random", max_iter=10000, random_state=0)
        nmf.fit(faces.reshape(len(faces), -1))

        errors.append(nmf.reconstruction_err_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 51), errors, marker="o")
    plt.xticks([1, 10, 20, 30, 40, 50], [1, 10, 20, 30, 40, 50])
    plt.title("NMF Reconstruction Error")
    plt.xlabel("number of components")
    plt.ylabel("reconstruction error")
    plt.show()


if __name__ == "__main__":
    nmf_face()

    nmf_reconstruction_error()

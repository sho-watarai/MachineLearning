import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

np.random.seed(0)


def kmeans_clustering():
    #
    # make dataset
    #
    X, _ = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

    #
    # k-means
    #
    kms = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, tol=1e-04, random_state=0)
    kms.fit(X)

    y_pred = kms.predict(X)

    #
    # visualization
    #
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], s=50, label="cluster %d" % (i + 1))
    plt.scatter(kms.cluster_centers_[:, 0], kms.cluster_centers_[:, 1], s=50, marker="x", c="red", label="centroid")
    plt.title("K-means clustering")
    plt.legend()
    plt.show()


def kmeans_elbow():
    instance_file = "../../../ComputerVision/COCO/COCO/annotations/instances_train2014.json"

    with open(instance_file, "rb") as file:
        dataset = json.load(file)

    images = dataset["images"]
    annotations = dataset["annotations"]

    image_dict = {}
    for im in images:
        image_dict[im["id"]] = im["height"], im["width"]

    bbox_dict = {}
    for ann in annotations:
        image_id = ann["image_id"]
        bbox = ann["bbox"]

        bbox_dict.setdefault(image_id, []).append(bbox)

    bounding_boxes = np.zeros((len(annotations), 2), dtype="float32")

    num_bboxes = 0
    for image_id, bbox_list in bbox_dict.items():
        height, width = image_dict[image_id]

        for bbox in bbox_list:
            box = [(bbox[0] + bbox[2] / 2) / width, (bbox[1] + bbox[3] / 2) / height, bbox[2] / width, bbox[3] / height]

            bounding_boxes[num_bboxes, :] = box[2:]
            num_bboxes += 1

    #
    # elbow method
    #
    distortions = []
    for i in range(1, 11):
        kms = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
        kms.fit(bounding_boxes)

        distortions.append(kms.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), distortions, marker="o")
    plt.xticks([i for i in range(1, 11)], [i for i in range(1, 11)])
    plt.title("Elbow method")
    plt.xlabel("number of clusters")
    plt.ylabel("distortion")
    plt.show()


if __name__ == "__main__":
    kmeans_clustering()

    kmeans_elbow()
    

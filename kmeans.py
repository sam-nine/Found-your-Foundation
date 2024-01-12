import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def silhouette_method(data, k_max=10):
    silhouette_scores = []

    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_k = np.argmax(silhouette_scores) + 2  # Adding 2 because we started from k=2
    return optimal_k

input_directory = "extracted_faces"
output_directory = "segmented_faces"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_directory, filename)

        # Read the image
        image = cv2.imread(image_path)
        
        # Downsample the image
        downsampled_image = cv2.resize(image, (100, 100))  # Adjust the size as needed

        hsv_image = cv2.cvtColor(downsampled_image, cv2.COLOR_BGR2LAB)
        pixels = hsv_image.reshape((-1, 3))

        # Determine optimal k using the silhouette method
        optimal_k = silhouette_method(pixels)
        print(optimal_k)

        # Perform k-means clustering with the optimal k
        kmeans = KMeans(n_clusters=optimal_k)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        segmented_image = centers[labels].reshape(hsv_image.shape)
        segmented_image_rgb = cv2.cvtColor(np.uint8(segmented_image), cv2.COLOR_LAB2BGR)

        output_path = os.path.join(output_directory, f"segmented_{filename}")
        cv2.imwrite(output_path, segmented_image_rgb)

        cv2.imshow("Original Image", image)
        cv2.imshow("Downsampled Image", downsampled_image)
        cv2.imshow("LAB Image", hsv_image)
        cv2.imshow("Segmented Image", segmented_image_rgb)
        cv2.waitKey(0)

cv2.destroyAllWindows()

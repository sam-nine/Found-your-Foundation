import cv2
import os
import numpy as np
from sklearn.cluster import KMeans


input_directory = "extracted_faces"
output_directory = "segmented_faces"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

k = 3

for filename in os.listdir(input_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_directory, filename)
        

        image = cv2.imread(image_path)
        

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


        pixels = hsv_image.reshape((-1, 3))


        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)


        labels = kmeans.labels_
        centers = kmeans.cluster_centers_


        segmented_image = centers[labels].reshape(hsv_image.shape)


        segmented_image_rgb = cv2.cvtColor(np.uint8(segmented_image), cv2.COLOR_LAB2RGB)


        output_path = os.path.join(output_directory, f"segmented_{filename}")
        cv2.imwrite(output_path, segmented_image_rgb)

        cv2.imshow("Original Image", image)
        cv2.imshow("LAB Image", hsv_image)
        cv2.imshow("Segmented Image", segmented_image_rgb)	
        cv2.waitKey(0)


cv2.destroyAllWindows()

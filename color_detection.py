import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import webcolors

# ----- Step 1: Define color dictionary -----
COLOR_NAMES = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue' :(0, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128)
}
# Step 2: Find closest color name from custom dictionary
def closest_color_name(rgb):
    min_dist = float('inf')
    closest_name = None
    for name, rgb_val in COLOR_NAMES.items():
        dist = distance.euclidean(rgb, rgb_val)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# Step 3: Find closest CSS3 color name using webcolors
def get_web_color_name(rgb_tuple):
    min_dist = float('inf')
    closest_name = None
    for name, hex_val in webcolors.CSS3_NAMES_TO_HEX.items():
        r, g, b = webcolors.hex_to_rgb(hex_val)
        dist = (r - rgb_tuple[0])**2 + (g - rgb_tuple[1])**2 + (b - rgb_tuple[2])**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


# OPTIONAL helper — returns closest color by RGB distance
def get_closest_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple)  # exact match
    except:
        return get_web_color_name(rgb_tuple)     # fallback to closest color


# Step 4: Detect dominant color using KMeans
def get_dominant_color(image, k=3):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.float32(image_rgb.reshape(-1, 3))

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(pixels)

    # Find most common cluster
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    return tuple(int(c) for c in dominant)


# Step 5: Main testing
frame = cv2.imread("my_image.jpg")  # Replace with actual image path

if frame is None:
    raise ValueError("Image not found. Check your file path.")

# Example bounding box (x, y, width, height)
x, y, w, h = 100, 100, 200, 200
cropped_img = frame[y:y+h, x:x+w]

dominant_rgb = get_dominant_color(cropped_img)
web_color_name = get_web_color_name(dominant_rgb)
closest_name = get_closest_color_name(dominant_rgb)

print(f"Dominant RGB: {dominant_rgb}")
print(f"Closest CSS3 color: {web_color_name}")
print(f"Color Name (fallback): {closest_name}")

cv2.imshow("Cropped Object", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import os
print(os.getcwd())  # Correct way to print current working directory
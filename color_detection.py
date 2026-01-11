def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = image.reshape(-1, 3)

    # Remove extreme values
    pixels = pixels[(pixels[:,0] > 20) & (pixels[:,0] < 240)]

    if len(pixels) > 1000:
        pixels = pixels[np.random.choice(len(pixels), 1000, replace=False)]

    kmeans = KMeans(n_clusters=2, n_init=5)
    kmeans.fit(pixels)

    dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    # Convert back to RGB
    dominant_rgb = cv2.cvtColor(
        np.uint8([[dominant]]),
        cv2.COLOR_LAB2RGB
    )[0][0]

    return tuple(int(c) for c in dominant_rgb)

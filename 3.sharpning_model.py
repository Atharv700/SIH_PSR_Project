import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial import distance

def track_black_regions(image, min_area_factor=40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min([cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 0])
    tracked_regions = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area * min_area_factor]
    return tracked_regions

def find_center(region):
    M = cv2.moments(region)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def apply_gradient_color(image_path, color_data, regions, output_image_path):
    image = cv2.imread(image_path)
    color_data = np.array([list(map(int, color)) for color in color_data])

    for i, region in enumerate(regions):
        center = find_center(region)
        if center is None:
            continue

        mask = np.zeros_like(image[:, :, 0])
        cv2.drawContours(mask, [region], -1, 255, -1)
        
        points = np.column_stack(np.where(mask == 255))
        
        distances = distance.cdist(points, [center], 'euclidean').flatten()
        sorted_indices = np.argsort(distances)[::-1]
        
        max_dist = distances.max()
        num_color_steps = len(color_data)

        for idx, point in enumerate(points[sorted_indices]):
            x, y = point
            color_index = min(int((distances[idx] / max_dist) * num_color_steps), num_color_steps - 1)
            image[x, y] = color_data[color_index]

    cv2.imwrite(output_image_path, image)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Enhanced Image with Improved Coloring")
    plt.axis('off')
    plt.show()

psr_image_path = r'C:\Users\dnyan\OneDrive\Desktop\Hack24\3.sharpning_model\ii1.jpg'  # Update with actual path
psr_image = cv2.imread(psr_image_path)

tracked_regions = track_black_regions(psr_image, min_area_factor=40)

color_data_path = r'C:\Users\dnyan\OneDrive\Desktop\Hack24\3.sharpning_model\rgb1.json'  # Update with actual path to JSON file
output_image_path = 'enhanced_psr_image.jpg'  # Output file path

with open(color_data_path, 'r') as f:
    color_data = json.load(f)

apply_gradient_color(psr_image_path, color_data, tracked_regions, output_image_path)











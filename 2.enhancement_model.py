import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

def find_adjacent_color(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    non_zero_pixels = np.where(mask == 255)
    if len(non_zero_pixels[0]) == 0:
        return (0, 255, 255)  # Default color if no adjacent pixels found
    
    colors = []
    for y, x in zip(non_zero_pixels[0], non_zero_pixels[1]):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    color = image[ny, nx]
                    if not np.array_equal(color, [0, 0, 0]):  # Avoid black
                        colors.append(color)
    
    if not colors:
        return (0, 255, 255)  # Default color if no adjacent colors found
    
    colors = np.array(colors)
    mean_color = np.mean(colors, axis=0).astype(int)
    
    mean_color = np.clip(mean_color, 0, 255)
    
    mean_color = np.clip(mean_color + 50, 0, 255)
    
    return tuple(mean_color)

def process_black_spots(image_path):
    img_cv = cv2.imread(image_path)
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    smallest_contour = None
    min_area = float('inf')
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area and area > 0:
            min_area = area
            smallest_contour = contour
    
    if smallest_contour is None:
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    processed_image = img_cv.copy()
    
    size_threshold = min_area * 30
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= size_threshold:
            fill_color = find_adjacent_color(img_cv, contour)
            
            if isinstance(fill_color, tuple) and len(fill_color) == 3:
                fill_color = tuple(map(int, fill_color))
                cv2.drawContours(processed_image, [contour], -1, fill_color, thickness=cv2.FILLED)
            else:
                cv2.drawContours(processed_image, [contour], -1, (0, 255, 255), thickness=cv2.FILLED)
    
    img_cv_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv_rgb)
    
    return img_pil

def load_image(file_path):
    processed_image = process_black_spots(file_path)
    
    processed_image = processed_image.resize((300, 300), Image.LANCZOS)
    
    img_tk = ImageTk.PhotoImage(processed_image)
    
    img_label.config(image=img_tk)
    img_label.image = img_tk

root = tk.Tk()
root.title("Black Spots with Adjacent Color Filling")

img_label = tk.Label(root)
img_label.pack()

# Correct file path to the image
image_path = r'C:\Users\dnyan\OneDrive\Desktop\Hack24\2.enhancement_model\ii1.jpg'  # Replace with your actual image file name and path

load_image(image_path)

root.mainloop()








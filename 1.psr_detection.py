import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


def detect_and_highlight_large_black_spots(image_path):
    img_cv = cv2.imread(image_path)
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # Detect dark spots
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    smallest_contour = None
    min_area = float('inf')
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area and area > 0:  # Avoid zero-area noise
            min_area = area
            smallest_contour = contour
    
    black_background = np.zeros_like(img_cv)
    
    size_threshold = min_area * 30
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= size_threshold:
            cv2.drawContours(black_background, [contour], -1, (255, 255, 0), thickness=cv2.FILLED)
    
    img_cv_rgb = cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv_rgb)
    
    return img_pil

def load_image(file_path):
    processed_image = detect_and_highlight_large_black_spots(file_path)
    
    processed_image = processed_image.resize((300, 300), Image.LANCZOS)
    
    img_tk = ImageTk.PhotoImage(processed_image)
    
    img_label.config(image=img_tk)
    img_label.image = img_tk

root = tk.Tk()
root.title("Highlighted Large Black Spots")

img_label = tk.Label(root)
img_label.pack()

image_path = r"C:\Users\dnyan\OneDrive\Desktop\Hack24\1.psr_detection\ii1.jpg"  

load_image(image_path)
root.mainloop()




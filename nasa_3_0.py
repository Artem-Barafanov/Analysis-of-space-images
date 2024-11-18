import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import multiprocessing as mp
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Глобальные переменные
file_paths = []

def process_image_segment(image, segment_number, queue, output_directory):
    image_with_objects = image.copy()
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    space_objects = []

    font = ImageFont.truetype("arial.ttf", 14)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + width / 2 
        center_y = y + height / 2
        brightness = np.sum(gray_image[y:y + height, x:x + width])
        object_type = classify_object(area, brightness)
        space_object = {
            "x": center_x,
            "y": center_y,
            "brightness": brightness,
            "type": object_type,
            "size": width * height }
        space_objects.append(space_object)
        cv2.rectangle(image_with_objects, (x, y), (x + width, y + height), (255, 0, 255), 2)
        pil_image = Image.fromarray(image_with_objects)
        draw = ImageDraw.Draw(pil_image)
        draw.text((x, y - 15), object_type, font=font, fill=(0, 0, 0))
        image_with_objects = np.array(pil_image)

    os.makedirs(output_directory, exist_ok=True)
    output2_directory = os.path.join(output_directory, "image_crop")
    os.makedirs(output2_directory, exist_ok=True)
    cv2.imwrite(os.path.join(output2_directory, f"{segment_number}.tif"), image_with_objects)

    with open(os.path.join(output_directory, f"{segment_number}.txt"), "w", encoding="utf-8") as file:
        for obj in space_objects:
            file.write(f"Координаты: ({obj['x']}, {obj['y']}); Яркость: {obj['brightness']}; Размер: {obj['size']}; Тип: {obj['type']}\n")
    print(f"Выполнен процесс №{segment_number}")
    queue.put((image_with_objects, segment_number - 1))
    return

def classify_object(area, brightness):
    return {
        area < 10 and brightness > 100: "звезда",
        area < 10 and brightness > 50: "комета",
        area < 10 and brightness > 0: "планета",
        area > 10000 and brightness > 1000000: "галактика",
        area < 10000 and brightness > 1000000: "квазар",
        area >= 10 and brightness > 0: "звезда"
    }[True]

def divide_image_into_segments(image, num_parts):
    height, width, _ = image.shape 
    part_width = (width // num_parts) + 1
    part_height = (height // num_parts) + 1
    parts = []
    for chunk_width in range(num_parts):
        for chunk_height in range(num_parts):
            part = image[chunk_height * part_height:min((chunk_height + 1) * part_height, len(image))]
            part = part[:, chunk_width * part_width:(chunk_width + 1) * part_width, :]
            parts.append(part)
    return parts

def process_images_in_parallel(image_paths, progress_var, progress_bar):
    total_images = len(image_paths)
    completed = 0

    for full_path_to_image in image_paths:
        directory = r'C:\Users\sibfl\Images'
        queue = mp.Queue()
        image = cv2.imread(full_path_to_image)
        num_parts = 4 
        mp_parts = divide_image_into_segments(image, num_parts)

        processes = []
        segment_number = 0 
        output_directory = os.path.join("image_result", os.path.splitext(os.path.basename(full_path_to_image))[0])
        for mp_part in mp_parts:
            segment_number += 1 
            process = mp.Process(target=process_image_segment, args=(mp_part, segment_number, queue, output_directory))
            process.start()
            processes.append(process)

        sum_finish = 0
        image_parts = [0] * len(mp_parts)

        while sum_finish != len(processes):
            if not queue.empty():
                sum_finish += 1
                image_part = queue.get()
                image_parts[image_part[1]] = image_part[0].copy()

        image_vstack = [image_parts[i] for i in range(0, num_parts ** 2, num_parts)]

        k = 0
        for i in range(num_parts):
            for j in range(1, num_parts):
                image_vstack[i] = np.vstack([image_vstack[i], image_parts[j + k]])
            k += num_parts

        image_with_objects = np.hstack(image_vstack)

        new_directory = directory[:directory.rfind("/")]
        cv2.imwrite(os.path.join(output_directory, "new_image.tif"), image_with_objects)

        completed += 1 
        progress_var.set((completed / total_images) * 100)
        progress_bar.update()

    messagebox.showinfo("Готово", "Результат сохранен")

def choose_images():
    global file_paths
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.tif;*.jpg;*.png")])
    messagebox.showinfo("Готово", "Изображения успешно загружены")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("КОСМОАНАЛИЗАТОР БАРАФАНОВА")
    root.geometry("600x400")

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12))
    style.configure("TLabel", font=("Arial", 10))

    instruction_frame = ttk.Frame(root)
    instruction_frame.pack(pady=10)

    button_frame = ttk.Frame(root)
    button_frame.pack(pady=20)

    choose_button = ttk.Button(button_frame, text="Загружать", command=choose_images)
    choose_button.grid(row=0, column=0, padx=10)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate", variable=progress_var)
    progress_bar.pack(pady=10)

    start_button = ttk.Button(button_frame, text="Анализировать", command=lambda: process_images_in_parallel(file_paths, progress_var, progress_bar))
    start_button.grid(row=0, column=1, padx=10)

    root.mainloop()
    
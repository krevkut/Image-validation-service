import os
import sys
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from deepface import DeepFace
from tkinter import filedialog, messagebox

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    deepface_home = os.path.join(base_path, ".deepface")
    keras_home = deepface_home
    db_path = os.path.join(base_path, "celeb_db")
else:
    home = os.path.expanduser("~")
    deepface_home = os.path.join(home, ".deepface")
    keras_home = os.path.join(home, ".keras")
    db_path = "D:/UNI/PROEKT UCHEBNY/celeb_db"

os.environ["DEEPFACE_HOME"] = deepface_home
os.environ["KERAS_HOME"] = keras_home

if getattr(sys, 'frozen', False):
    cv2_base_dir = os.path.join(sys._MEIPASS, "cv2")
    haar_path = os.path.join(cv2_base_dir, "data")
    cv2.data.haarcascades = haar_path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if getattr(sys, 'frozen', False):
    deepface_weights_path = os.path.join(sys._MEIPASS, "weights", ".deepface")
    os.environ["DEEPFACE_HOME"] = deepface_weights_path
    os.environ["KERAS_HOME"] = deepface_weights_path

if getattr(sys, 'frozen', False):
    CELEB_DB_PATH = os.path.join(sys._MEIPASS, "celeb_db")
else:
    CELEB_DB_PATH = "D:/UNI/PROEKT UCHEBNY/celeb_db"

representations_path = os.path.join(CELEB_DB_PATH, "ds_model_arcface_detector_opencv_aligned_normalization_base_expand_0.pkl")
if not os.path.exists(representations_path):
    print("⚡ Создаю кеш для базы знаменитостей...")
    DeepFace.find(img_path=os.path.join(CELEB_DB_PATH, os.listdir(CELEB_DB_PATH)[0]), db_path=CELEB_DB_PATH, enforce_detection=True)
    print("✅ Кеш создан!")

def is_celebrity(image_path, threshold=0.4):
    try:
        result = DeepFace.find(img_path=image_path, db_path=CELEB_DB_PATH, enforce_detection=True, distance_metric="cosine", model_name="ArcFace")
        result_df = result[0]

        if result_df.empty:
            return False

        min_distance = result_df.iloc[0]["distance"]
        print(f"Минимальное расстояние: {min_distance:.4f}")

        return min_distance < threshold

    except Exception as e:
        print("Ошибка при анализе лица:", e)
        return False


def upload_image():
    global selected_img_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])

    if not file_path:
        return

    selected_img_path = file_path
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img)

    image_label.configure(image=img_tk)
    image_label.image = img_tk
    status_label.configure(text="Выберите 'Подтвердить' для проверки", text_color="yellow")
    confirm_button.pack(pady=10)


def analyze_image():
    global selected_img_path

    if not selected_img_path:
        messagebox.showerror("Ошибка", "Сначала загрузите изображение!")
        return

    status_label.configure(text="Обрабатываю фото...", text_color="yellow")

    if is_celebrity(selected_img_path):
        status_label.configure(text="Обнаружена знаменитость!", text_color="red")
        messagebox.showwarning("Результат", "Обнаружено лицо знаменитости! Доступ запрещен.")
    else:
        status_label.configure(text="Лицо не является знаменитостью", text_color="green")
        messagebox.showinfo("Результат", "Лицо не является знаменитостью.")


root = ctk.CTk()
root.title("ИИ-проверка лица")
root.wm_attributes('-alpha', 0.97)
root.geometry("500x500")

title_label = ctk.CTkLabel(root, text="Проверка лица ИИ", font=("Arial", 20, "bold"))
title_label.pack(pady=10)

btn_upload = ctk.CTkButton(root, text="Выбрать фото", command=upload_image)
btn_upload.pack(pady=10)

image_label = ctk.CTkLabel(root, text="")
image_label.pack(pady=10)

status_label = ctk.CTkLabel(root, text="Ожидание загрузки фото...", font=("Arial", 14))
status_label.pack(pady=10)

confirm_button = ctk.CTkButton(root, text="Подтвердить", command=analyze_image, fg_color="green")
confirm_button.pack_forget()

root.mainloop()

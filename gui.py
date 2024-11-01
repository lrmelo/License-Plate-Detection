# gui.py

import torch
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from torchvision import transforms as T
from model import get_model

# Definir o dispositivo
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Número de classes (1 classe + background)
num_classes = 2

# Obter o modelo
model = get_model(num_classes)
model.load_state_dict(torch.load('modelo_placas.pth', map_location=device))
model.to(device)
model.eval()

# Transformação
transform = T.ToTensor()

def load_image():
    global img, img_tk, img_tensor
    filepath = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", ("*.jpg", "*.jpeg", "*.png")),
            ("All Files", "*.*")
        ]
    )
    if filepath:
        img = Image.open(filepath).convert("RGB")
        img_resized = img.resize((500, 400))  # Redimensionar para caber na interface
        img_tk = ImageTk.PhotoImage(img_resized)
        img_tensor = transform(img).to(device)

        # Atualizar a imagem no label
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Limpar o resultado anterior
        result_label.config(text="")


def run_inference():
    global img, img_tensor
    if img_tensor is not None:
        with torch.no_grad():
            prediction = model([img_tensor])

        # Processar as detecções
        boxes = prediction[0]['boxes'].cpu()
        scores = prediction[0]['scores'].cpu()

        # Definir um threshold
        threshold = 0.5
        has_plate = False
        for score in scores:
            if score > threshold:
                has_plate = True
                break

        if has_plate:
            result_label.config(text="Placa detectada na imagem.", fg='green')
            # Exibir a imagem com as caixas
            show_image_with_boxes(img, boxes, scores, threshold)
        else:
            result_label.config(text="Nenhuma placa detectada.", fg='red')
    else:
        result_label.config(text="Por favor, carregue uma imagem primeiro.", fg='red')

def show_image_with_boxes(img, boxes, scores, threshold):
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    for box, score in zip(boxes, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
            draw.text((xmin, ymin - 10), f"{score:.2f}", fill="red")

    # Atualizar a imagem na interface
    img_resized = img_with_boxes.resize((500, 400))
    img_tk_with_boxes = ImageTk.PhotoImage(img_resized)
    image_label.config(image=img_tk_with_boxes)
    image_label.image = img_tk_with_boxes

# Inicializar a janela principal
root = tk.Tk()
root.title("Detecção de Placas")
root.geometry("600x700")

# Variáveis globais
img = None
img_tk = None
img_tensor = None

# Frame para os botões
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Botão para carregar a imagem
load_button = tk.Button(button_frame, text="Carregar Imagem", command=load_image)
load_button.pack(side=tk.LEFT, padx=5)

# Botão para executar a inferência
infer_button = tk.Button(button_frame, text="Detectar Placa", command=run_inference)
infer_button.pack(side=tk.LEFT, padx=5)

# Label para exibir a imagem
image_label = tk.Label(root)
image_label.pack()

# Label para exibir o resultado
result_label = tk.Label(root, text="", font=('Arial', 14))
result_label.pack(pady=10)

# Iniciar o loop principal
root.mainloop()

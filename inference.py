#Script para realizar inferências com o modelo treinado.
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
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

# Função para exibir a imagem com as detecções
def show_predictions(image, predictions, threshold=0.5):
    plt.figure(figsize=(12, 8))
    img = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img)
    ax = plt.gca()

    boxes = predictions[0]['boxes'].detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy()

    for box, score in zip(boxes, scores):
        if score > threshold:
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            rect = plt.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{score:.2f}", color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.show()

# Carregar uma imagem de teste
image_path = '/media/lucas/SSD Secundario/upf/trabfinal2/complaca2.png'  # Substitua pelo caminho da sua imagem
image = Image.open(image_path).convert("RGB")
transform = T.ToTensor()
image_tensor = transform(image).to(device)

# Fazer a previsão
with torch.no_grad():
    prediction = model([image_tensor])

# Exibir a imagem com as detecções
show_predictions(image_tensor, prediction, threshold=0.5)

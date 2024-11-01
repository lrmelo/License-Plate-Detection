import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
#Contém a classe LicensePlateDataset que cria o dataset personalizado.
class LicensePlateDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # Obter todas as imagens e ordená-las
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))

        # Obter todas as anotações e ordená-las
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # Carregar a imagem
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Carregar a anotação
        annot_path = os.path.join(self.root, "annotations", self.annotations[idx])
        tree = ET.parse(annot_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for member in root.findall('object'):
            label = member.find('name').text
            if label != 'licence':
                continue  # Ignorar outras classes, se houver


            bndbox = member.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Usamos '1' para representar a classe 'placa'

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Criar o dicionário de alvos (targets)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # Outras informações (opcionais)
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Campos obrigatórios para modelos do torchvision
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

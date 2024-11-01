#Script principal que realiza o treinamento.
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import LicensePlateDataset
from transforms import get_transform
from model import get_model
from utils import collate_fn

# Definir o dispositivo
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Número de classes (1 classe + background)
num_classes = 2

# Obter o modelo
model = get_model(num_classes)
model.to(device)

# Criar o dataset
dataset = LicensePlateDataset('dataset', get_transform(train=True))
dataset_test = LicensePlateDataset('dataset', get_transform(train=False))

# Dividir o dataset em treinamento e validação
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
num_train = int(0.8 * len(indices))
dataset = torch.utils.data.Subset(dataset, indices[:num_train])
dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:])

# Definir os DataLoaders
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
test_loader = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Parâmetros que serão otimizados
params = [p for p in model.parameters() if p.requires_grad]

# Definir o otimizador
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Agendador de taxa de aprendizado (opcional)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in tqdm(train_loader, desc=f"Treinando época {epoch+1}/{num_epochs}"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Computar a perda
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Retropropagação
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        i += 1

    # Atualizar o agendador de taxa de aprendizado
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Época {epoch+1} finalizada.")

print("Treinamento concluído.")

# Salvar o modelo treinado
torch.save(model.state_dict(), 'modelo_placas.pth')

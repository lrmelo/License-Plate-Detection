import torchvision.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # Adicione aqui outras transformações de data augmentation se desejar
        pass
    return T.Compose(transforms)

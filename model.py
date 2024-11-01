#Contém a função para carregar e modificar o modelo pré-treinado.
import torchvision

def get_model(num_classes):
    # Carregar o modelo pré-treinado
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Obter o número de entradas do classificador
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Substituir o classificador pré-treinado por um novo
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

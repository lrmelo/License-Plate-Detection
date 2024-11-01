# Projeto de Detecção de Placas de Carro
## Introdução
Este projeto implementa um modelo de detecção de placas de carros utilizando PyTorch e torchvision. O modelo é treinado para identificar placas em imagens e pode ser utilizado através de uma interface gráfica simples construída com Tkinter.
## Dependências
- `Python 3.x` (recomendado Python 3.10)
- `pip` (gerenciador de pacotes do Python)
- `PyTorch` e `torchvision`
- `Pillow` (biblioteca de imagens)
- `Matplotlib` (para visualização de imagens)
- `Tkinter` (interface gráfica padrão do Python)
- `tqdm` (barra de progresso)

## Instalação das Dependências
Atualizar o pip
```
pip install --upgrade pip
```

Instalar as bibliotecas necessárias
```
pip install torch torchvision
pip install Pillow
pip install matplotlib
pip install tqdm
...
```

(opcional) Caso ocorra algum erro de dependencia, crie um ambiente virtual para isolar as dependências do projeto. Siga os passos abaixo:

Linux

```
python3 -m venv env
source env/bin/activate
```


## Estrutura do Projeto

O projeto é composto pelos seguintes arquivos:

- `dataset.py`: Contém a classe `LicensePlateDataset`, responsável por carregar as imagens e anotações e preparar os dados para o treinamento.

- `model.py`: Define a função `get_model`, que carrega o modelo pré-treinado e ajusta o número de classes.

- `utils.py`: Contém funções auxiliares, como `collate_fn`, utilizada no DataLoader.

- `transforms.py`: Define as transformações aplicadas às imagens, incluindo data augmentation.

- `train.py`: Script principal para treinar o modelo. Realiza o treinamento utilizando os dados fornecidos e salva o modelo treinado.

- `inference.py`: Permite fazer inferências utilizando o modelo treinado em imagens específicas e visualizar os resultados.

- `gui.py`: Implementa uma interface gráfica com Tkinter, permitindo ao usuário selecionar uma imagem e verificar se há uma placa detectada pelo modelo.

- `modelo_placas.pth`: Arquivo gerado após o treinamento que contém os pesos do modelo treinado.

- `dataset/`: Diretório que contém as imagens e anotações utilizadas para treinar e testar o modelo.

    - `dataset/images/`: Contém as imagens.
    - `dataset/annotations/`: Contém as anotações em formato XML.

## Instruções de Uso

Para treinar o modelo, execute o script `train.py`. 
```
python train.py
```

Caso queira evitar utilizar a interface grafica edite o arquivo `inference.py` e ajuste o caminho da imagem que deseja testar
```
image_path = 'caminho/para/sua/imagem.jpg'  # Substitua pelo caminho da sua imagem
```
E execute 
```
python inference.py
```
### Uso da Interface Gráfica
Para utilizar a interface gráfica que permite selecionar imagens e verificar se há placas detectadas, execute o script `gui.py`.
```
python gui.py
```
A aplicação abrirá uma janela com os seguintes componentes:

- Botão "Carregar Imagem": Permite selecionar uma imagem do seu sistema de arquivos.
- Área de Exibição da Imagem: Mostra a imagem selecionada.
- Botão "Detectar Placa": Executa a inferência usando o modelo treinado.
- Mensagem de Resultado: Indica se uma placa foi detectada ou não.
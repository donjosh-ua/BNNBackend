import torchvision.datasets as dset
import torchvision
from app.bnn.redneuronal_bay.RedNeuBay import RedNeuBay
from app.bnn.redneuronal_bay.Layers.layers import *
from app.bnn.redneuronal_bay.preprocesamiento import *
from app.bnn.redneuronal_bay.metricas_eva import *
from app.bnn.redneuronal_bay.funcion_activacion import *

import pandas as pd

filename = "./app/data/pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
df_cla = pd.read_csv(filename, names=names)  # Base de datos tipo data frame


def train(alpha=0.001, epoch=20, criteria="cross_entropy", optimizer="SGD", image_size=None, 
          verbose=True, decay=0.0, momentum=0.9, image=False, FA_ext=None, Bay=False, 
          save_mod="ModiR", pred_hot=True, test_size=0.2, batch_size=64, cv=True, Kfold=5):
    """
    Entrenamiento de la red neuronal
    
    Args:
        alpha (float): Learning rate
        epoch (int): Number of training epochs
        criteria (str): Loss function
        optimizer (str): Optimizer type
        image_size (int, optional): Image size for CNN
        verbose (bool): Verbose output
        decay (float): Weight decay
        momentum (float): Momentum for SGD
        image (bool): Whether input is image data
        FA_ext (str, optional): External activation function
        Bay (bool): Use Bayesian neural network
        save_mod (str): Model save name
        pred_hot (bool): Use one-hot prediction
        test_size (float): Test set ratio
        batch_size (int): Batch size
        cv (bool): Use cross-validation
        Kfold (int): Number of folds for cross-validation
    """
    Red_Bay = RedNeuBay(
        alpha=alpha,
        epoch=epoch,
        criteria=criteria,
        optimizer=optimizer,
        image_size=image_size,
        verbose=verbose,
        decay=decay,
        momentum=momentum,
        image=image,
        FA_ext=FA_ext,
        Bay=Bay,
        save_mod=save_mod,
        pred_hot=pred_hot,
        test_size=test_size,
        batch_size=batch_size,
        cv=cv,
        Kfold=Kfold,
    )

    Red_Bay.add(Tanh_Layer(8, 13))  # Capa de entrada
    Red_Bay.add(Tanh_Layer(13, 8))  # Capa oculta
    # Red_Bay.add(SoftmaxBay_Layer(10, 2))  # Capa final bayesiana
    Red_Bay.add(Softmax_Layer(8, 2))
    # Red_Bay.add(Sigmoid_Layer(8, 1))  # Capa final
    # Si deseara aplicar una funcion exttra ala salida de las capas por ejemplo una softmax - colocar en funcion
    print(Red_Bay)
    # out = Red_Bay.train(df_cla=df_cla) #Sin cross validacion
    out = Red_Bay.cv_train(df_cla=df_cla)  # Con cross validacion
    return out


# # torch.manual_seed(123) #fijamos la semilla
# transforms = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.1307,), (0.3081,)),
#     ]
# )

# root = "./app/data"
# train_set = dset.MNIST(root=root, train=True, transform=transforms, download=False)
# test_set = dset.MNIST(root=root, train=False, transform=transforms)

# Red 1    image_size=784 porque imagen es de 28x28

# Modelo  Tanh y Softmax
# ------------------------SIN BAYESIANO VALIDACION 50 epochs--------------------------------
# Red_Bay = RedNeuBay(
#     alpha=0.001,
#     epoch=50,
#     criteria="cross_entropy",
#     optimizer="SGD",
#     image_size=784,
#     verbose=True,
#     decay=0.0,
#     momentum=0.9,
#     image=True,
#     FA_ext=None,
#     Bay=False,
#     save_mod="Img_ori2",
#     pred_hot=True,
#     test_size=None,
#     batch_size=64,
#     cv=False,
#     Kfold=5,
# )

# Red_Bay.add(Tanh_Layer(784, 1000))  # Capa de entrada
# Red_Bay.add(Tanh_Layer(1000, 50))  # Capa oculta
# Red_Bay.add(SoftmaxBay_Layer(50,10))     #Capa final
# Red_Bay.add(Softmax_Layer(50, 10))

# print(Red_Bay)
# out = Red_Bay.train(train_set=train_set, test_set=test_set)
# out

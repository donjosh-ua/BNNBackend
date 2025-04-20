from app.bnn.redneuronal_bay.RedNeuBay import RedNeuBay
from app.bnn.redneuronal_bay.Layers.layers import *
from app.bnn.redneuronal_bay.preprocesamiento import *
from app.bnn.redneuronal_bay.metricas_eva import *
from app.bnn.redneuronal_bay.funcion_activacion import *
import re
from typing import List, Tuple
from app.config import config

import pandas as pd

filename = config.get("data_file")
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
df_cla = pd.read_csv(filename, names=names)  # Base de datos tipo data frame


def parse_layer_spec(layer_spec: str) -> Tuple[str, Tuple[int, int]]:
    """
    Parse a layer specification string.
    
    Example: "Tanh(10, 80)" -> ("Tanh", (10, 80))
    Example: "\"Tanh(10, 80)\"" -> ("Tanh", (10, 80))
    
    Args:
        layer_spec (str): Layer specification in the format "Activation(inputs, outputs)"
                         or "\"Activation(inputs, outputs)\""
    
    Returns:
        Tuple[str, Tuple[int, int]]: Tuple containing the activation function name and input/output dimensions
    """
    # Remove quotes if they exist
    layer_spec = layer_spec.strip('"\'')
    
    # Use regex to extract activation name and parameters
    match = re.match(r'(\w+)\((\d+),\s*(\d+)\)', layer_spec)
    if not match:
        raise ValueError(f"Invalid layer format: {layer_spec}. Expected format: Activation(inputs, outputs)")
    
    activation_name = match.group(1)
    inputs = int(match.group(2))
    outputs = int(match.group(3))
    
    return activation_name, (inputs, outputs)


def create_layer(layer_spec: str, bay: bool = False):
    """
    Factory function to create a layer from a specification string.
    
    Args:
        layer_spec (str): Layer specification in the format "Activation(inputs, outputs)"
        bay (bool): Whether to use Bayesian layers when available
    
    Returns:
        Layer: The created layer object
    """
    activation_name, params = parse_layer_spec(layer_spec)
    inputs, outputs = params
    
    # Map of activation names to layer classes
    if activation_name == "Tanh":
        return Tanh_Layer(inputs, outputs)
    elif activation_name == "Softmax":
        return Softmax_Layer(inputs, outputs)
    elif activation_name == "SoftmaxBay" and bay:
        return SoftmaxBay_Layer(inputs, outputs)
    elif activation_name == "Sigmoid":
        return Sigmoid_Layer(inputs, outputs)
    elif activation_name == "ReLU":
        return ReLU_Layer(inputs, outputs)
    elif activation_name == "LeakyReLU":
        return LeakyReLU_Layer(inputs, outputs)
    elif activation_name == "LogSoftmax":
        return Log_Softmax_Layer(inputs, outputs)
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")


def validate_layers(layers: List[str]):
    """
    Validate the layer specifications.
    
    Ensures:
    1. Layers are properly formatted
    2. Output dimensions of one layer match input dimensions of the next
    3. The activation functions are supported
    
    Args:
        layers (List[str]): List of layer specifications
    
    Raises:
        ValueError: If layers are invalid
    """
    valid_activations = ["Tanh", "Softmax", "SoftmaxBay", "Sigmoid", "ReLU"]
    
    for i, layer_spec in enumerate(layers):
        # Check format and extract activation name and dimensions
        try:
            activation_name, params = parse_layer_spec(layer_spec)
        except ValueError as e:
            raise ValueError(f"Invalid layer format: {e}")
            
        # Check activation name
        if activation_name not in valid_activations:
            raise ValueError(f"Unknown activation function: {activation_name}")
            
        # Check dimensions between layers
        if i > 0:
            try:
                _, prev_params = parse_layer_spec(layers[i-1])
                _, prev_outputs = prev_params
                inputs, _ = params
                
                if prev_outputs != inputs:
                    raise ValueError(f"Layer dimension mismatch: {layers[i-1]} outputs {prev_outputs}, but {layer_spec} expects {inputs} inputs")
            except Exception as e:
                raise ValueError(f"Error validating layer dimensions: {e}")


def train(alpha=0.001, epoch=20, criteria="cross_entropy", optimizer="SGD", image_size=None, 
          verbose=True, decay=0.0, momentum=0.9, image=False, FA_ext=None, Bay=False, 
          save_mod="ModiR", pred_hot=True, test_size=0.2, batch_size=64, cv=True, Kfold=5,
          layers=None):
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
        layers (List[str], optional): List of layer specifications in the format 
                                    "Activation(inputs, outputs)"
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

    # Use dynamic layers if specified, otherwise use default architecture
    if layers and len(layers) > 0:
        # Validate the layers before adding them
        validate_layers(layers)
        
        # Add each layer as specified
        for layer_spec in layers:
            layer = create_layer(layer_spec, Bay)
            Red_Bay.add(layer)
            
        print(f"Using custom architecture with {len(layers)} layers")
    else:
        # Use default architecture
        Red_Bay.add(Tanh_Layer(8, 13))  # Capa de entrada
        Red_Bay.add(Tanh_Layer(13, 8))  # Capa oculta
        Red_Bay.add(Softmax_Layer(8, 2))
        print("Using default architecture")

    print(Red_Bay)
    # Use cross validation or normal training based on parameters
    if cv:
        out = Red_Bay.cv_train(df_cla=df_cla)  # Con cross validacion
    else:
        out = Red_Bay.train(df_cla=df_cla)  # Sin cross validacion
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

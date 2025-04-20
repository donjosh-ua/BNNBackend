from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from app.bnn.RNA import train
from typing import Optional, List
import os
import json

router = APIRouter()

class NeuralNetworkParameters(BaseModel):
    alpha: float = Field(default=0.001, description="Learning rate")
    epoch: int = Field(default=20, description="Number of training epochs")
    criteria: str = Field(default="cross_entropy", description="Loss function")
    optimizer: str = Field(default="SGD", description="Optimizer type")
    image_size: Optional[int] = Field(default=None, description="Image size for CNN")
    verbose: bool = Field(default=True, description="Verbose output")
    decay: float = Field(default=0.0, description="Weight decay")
    momentum: float = Field(default=0.9, description="Momentum for SGD")
    image: bool = Field(default=False, description="Whether input is image data")
    FA_ext: Optional[str] = Field(default=None, description="External activation function")
    useBayesian: bool = Field(default=False, description="Use Bayesian neural network")
    save_mod: str = Field(default="ModiR", description="Model save name")
    pred_hot: bool = Field(default=True, description="Use one-hot prediction")
    test_size: float = Field(default=0.2, description="Test set ratio")
    batch_size: int = Field(default=64, description="Batch size")
    cv: bool = Field(default=True, description="Use cross-validation")
    numFolds: int = Field(default=5, description="Number of folds for cross-validation")
    layers: List[str] = Field(default=[], description="Neural network layers specifications in the form 'Activation_Function_Name(inputs, outputs)'")

def save_parameters_to_conf(params: dict):
    """
    Save neural network parameters to the configuration file.
    
    Args:
        params (dict): Dictionary containing the neural network parameters
    """
    config_path = "./app/config/nn_parameters.conf"
    with open(config_path, 'w') as f:
        f.write("{\n")
        # Write each parameter in Python literal format
        for i, (key, value) in enumerate(params.items()):
            if value is None:
                formatted_value = "None"
            elif isinstance(value, bool):
                formatted_value = str(value)
            elif isinstance(value, str):
                formatted_value = f'"{value}"'
            elif isinstance(value, list):
                if not value:  # Empty list
                    formatted_value = "[]"
                elif key == "layers":
                    # Format layers list with each element in quotes
                    layer_items = [f'"{layer}"' for layer in value]
                    formatted_value = f'[{", ".join(layer_items)}]'
                else:
                    formatted_value = str(value).replace("'", "")
            else:
                formatted_value = str(value)
            
            # Add comma for all items except the last one
            comma = "," if i < len(params) - 1 else ""
            f.write(f'    "{key}": {formatted_value}{comma}\n')
        
        f.write("}\n")

@router.post("/normal")
async def train_rna(params: NeuralNetworkParameters):
    """
    Entrena una red neuronal normal con los parámetros proporcionados.
    
    Los parámetros se envían desde el frontend y se pasan al método de entrenamiento.
    También se guardan en el archivo de configuración para uso futuro.
    """
    try:
        # Convert Pydantic model to dictionary and rename useBayesian to Bay to match API
        train_params = params.model_dump()
        
        # Save the original parameters to the configuration file
        save_parameters_to_conf(train_params)
        
        # Rename parameters for the train function
        train_params["Bay"] = train_params.pop("useBayesian")
        train_params["Kfold"] = train_params.pop("numFolds")
        
        # Call the train function with the parameters (including layers)
        train(**train_params)
        return {"message": "Entrenamiento de red neuronal completado con éxito y parámetros guardados."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

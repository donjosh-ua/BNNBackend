from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from app.bnn.RNA import train
from typing import Optional, List
import os
import json
import base64

router = APIRouter()


class BayesianConfig(BaseModel):
    distribution_type: str = Field(
        default="normal", description="Type of prior distribution"
    )
    mean: float = Field(default=0, description="Mean of the distribution")
    sigma: float = Field(default=1, description="Standard deviation or scale parameter")
    alpha: float = Field(
        default=0.1, description="Alpha parameter for the distribution"
    )
    beta: float = Field(default=0.1, description="Beta parameter for the distribution")
    lambdaPar: float = Field(
        default=0.1, description="Lambda parameter for the distribution"
    )


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
    FA_ext: Optional[str] = Field(
        default=None, description="External activation function"
    )
    useBayesian: bool = Field(default=False, description="Use Bayesian neural network")
    bayesian_config: Optional[BayesianConfig] = Field(
        default=None, description="Configuration for Bayesian priors"
    )
    save_mod: str = Field(default="ModiR", description="Model save name")
    pred_hot: bool = Field(default=True, description="Use one-hot prediction")
    test_size: float = Field(default=0.2, description="Test set ratio")
    batch_size: int = Field(default=64, description="Batch size")
    cv: bool = Field(default=True, description="Use cross-validation")
    numFolds: int = Field(default=5, description="Number of folds for cross-validation")
    layers: List[str] = Field(
        default=[],
        description="Neural network layers specifications in the form 'Activation_Function_Name(inputs, outputs)'",
    )


def save_parameters_to_conf(params: dict):
    """
    Save neural network parameters to the configuration file.

    Args:
        params (dict): Dictionary containing the neural network parameters
    """
    config_path = "./app/config/nn_parameters.conf"
    with open(config_path, "w") as f:
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
            elif isinstance(value, dict):
                # Handle nested dictionaries (like bayesian_config)
                formatted_items = []
                for k, v in value.items():
                    if isinstance(v, str):
                        formatted_items.append(f'"{k}": "{v}"')
                    else:
                        formatted_items.append(f'"{k}": {v}')
                formatted_value = "{" + ", ".join(formatted_items) + "}"
            else:
                formatted_value = str(value)

            # Add comma for all items except the last one
            comma = "," if i < len(params) - 1 else ""
            f.write(f'    "{key}": {formatted_value}{comma}\n')

        f.write("}\n")


@router.get("/results")
async def get_results():
    """
    Obtiene los resultados del entrenamiento de la red neuronal, incluyendo
    el JSON almacenado en results.json y los archivos de imagen (PNG) codificados en Base64.

    Returns:
        dict: Objeto con los datos JSON y las imágenes codificadas en base64.
    """
    try:
        # Load JSON results from file
        results_path = "./app/bnn/output/results.json"
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Results file not found.")
        with open(results_path, "r") as f:
            results_data = json.load(f)

        # Load regular PNG images from the plots folder
        images_folder = "./app/data/plots"
        images = {}
        if os.path.exists(images_folder):
            for file in os.listdir(images_folder):
                if file.endswith(".png"):
                    file_path = os.path.join(images_folder, file)
                    with open(file_path, "rb") as img_file:
                        encoded_image = base64.b64encode(img_file.read()).decode(
                            "utf-8"
                        )
                        images[file] = encoded_image

        # Load CV-specific images from the cv subfolder
        cv_images_folder = os.path.join(images_folder, "cv")
        if os.path.exists(cv_images_folder):
            for file in os.listdir(cv_images_folder):
                if file.endswith(".png"):
                    file_path = os.path.join(cv_images_folder, file)
                    with open(file_path, "rb") as img_file:
                        encoded_image = base64.b64encode(img_file.read()).decode(
                            "utf-8"
                        )
                        # Add to images with cv/ prefix to distinguish them
                        images[f"cv/{file}"] = encoded_image

        # Format all numerical results for better readability
        formatted_results = {}

        # Handle accuracy and epoch specially
        if "accuracy" in results_data:
            formatted_results["Accuracy"] = f"{results_data['accuracy']}%"
        if "epoch" in results_data:
            formatted_results["Best Epoch"] = str(results_data["epoch"])

        # Format class frequencies
        if "overall_class_frequency" in results_data:
            class_freq = []
            for class_name, count in results_data["overall_class_frequency"].items():
                class_freq.append(f"{class_name}: {count}")
            formatted_results["Class Frequency"] = ", ".join(class_freq)

        if "image_class_frequency" in results_data:
            img_class_freq = []
            for class_name, count in results_data["image_class_frequency"].items():
                img_class_freq.append(f"{class_name}: {count}")
            formatted_results["Image Class Frequency"] = ", ".join(img_class_freq)

        # Handle CV fold class frequencies
        if "class_frequency" in results_data:
            for fold, freqs in results_data["class_frequency"].items():
                fold_freqs = []
                for class_name, count in freqs.items():
                    fold_freqs.append(f"{class_name}: {count}")
                formatted_results[f"Class Frequency ({fold})"] = ", ".join(fold_freqs)

        # Return both the formatted results and encoded images
        return {
            "text_results": formatted_results,
            "raw_results": results_data,
            "images": images,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/normal")
async def train_rna(params: NeuralNetworkParameters):
    """
    Entrena una red neuronal normal con los parámetros proporcionados.

    Los parámetros se envían desde el frontend y se pasan al método de entrenamiento.
    También se guardan en el archivo de configuración para uso futuro.
    """
    try:
        # Convert Pydantic model to dictionary
        train_params = params.model_dump()

        # Extract bayesian config if available
        bayesian_config = train_params.pop("bayesian_config", None)

        # Rename parameters for the train function
        train_params["Bay"] = train_params.pop("useBayesian")
        train_params["Kfold"] = train_params.pop("numFolds")

        # Save the full config (including bayesian_config) to the conf file
        save_parameters = train_params.copy()
        if bayesian_config:
            save_parameters["bayesian_config"] = bayesian_config
        save_parameters_to_conf(save_parameters)

        # Add individual bayesian parameters to train_params if bayesian_config exists
        if bayesian_config:
            # Convert the BayesianConfig model to a dictionary if it's not already
            if not isinstance(bayesian_config, dict):
                bayesian_config = bayesian_config.dict()

            # Add each parameter individually
            train_params["distribution_type"] = bayesian_config.get(
                "distribution_type", "normal"
            )
            train_params["mean"] = bayesian_config.get("mean", 0.0)
            train_params["sigma"] = bayesian_config.get("sigma", 1.1)
            train_params["alpha"] = bayesian_config.get("alpha", 0.1)
            train_params["beta"] = bayesian_config.get("beta", 0.1)
            train_params["lambdaPar"] = bayesian_config.get("lambdaPar", 0.1)

        # Call the train function with the parameters (including layers)
        train(**train_params)
        return {
            "message": "Entrenamiento de red neuronal completado con éxito y parámetros guardados."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download")
async def download_best_model():
    """
    Endpoint to download the best model of the training
    Returns the best saved model as a downloadable file
    """
    try:
        # Get the model name from the results.json or use default
        parameters_path = "./app/config/nn_parameters.conf"
        model_name = "ModiR"  # Default model name

        if os.path.exists(parameters_path):
            try:
                with open(parameters_path, "r") as f:
                    nn_parameters = json.load(f)
                model_name = nn_parameters.get("save_mod", "ModiR")
            except:
                pass

        # Construct the best model filename
        best_model_filename = f"best_{model_name}"
        model_path = f"./app/bnn/output/{best_model_filename}"

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, detail=f"Model file not found: {best_model_filename}"
            )

        # Return the file as a download
        return FileResponse(
            path=model_path,
            filename=f"{best_model_filename}.pth",
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={best_model_filename}.pth"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error downloading model: {str(e)}"
        )

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.service.file_service import FileService
from typing import Dict, List

router = APIRouter()
file_service = FileService()

class DatasetSelection(BaseModel):
    file_path: str

@router.get("/list")
async def get_available_files():
    """
    Get all available data files categorized by type (csv, image)
    
    Returns:
        Dict with two keys: 'csv' and 'image', each containing a list of file paths
    """
    try:
        files = file_service.get_available_files()
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/select")
async def select_dataset(selection: DatasetSelection):
    """
    Select a dataset to be used for training and analysis
    
    Args:
        selection: An object containing the file_path to the selected dataset
        
    Returns:
        Dict with a success message and the selected file
    """
    try:
        result = file_service.set_selected_dataset(selection.file_path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/current")
async def get_current_dataset():
    """
    Get the currently selected dataset from settings.conf
    
    Returns:
        Dict with the current dataset file path
    """
    try:
        return file_service.get_current_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
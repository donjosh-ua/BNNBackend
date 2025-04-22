from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.service.file_service import FileService

import os

router = APIRouter()
file_service = FileService()


class DatasetSelection(BaseModel):
    file_path: str
    has_header: bool


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
    try:
        file_service.set_selected_dataset(selection.file_path, selection.has_header)
        preview = []
        # If the dataset is "mnist", don't send a preview.
        if selection.file_path.lower() == "mnist":
            preview = None
        else:
            # Get the directory of the current file (app/controller)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Check if the file_path already starts with "data" to avoid duplication
            if selection.file_path.startswith("data" + os.path.sep):
                # remove "data" folder prefix, we'll add it manually
                relative_path = selection.file_path[len("data" + os.path.sep):]
            else:
                relative_path = selection.file_path

            # Build the full path: (../data relative to app/controller)
            full_path = os.path.join(base_dir, "..", "data", relative_path)
            with open(full_path, 'r') as f:
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    preview.append(line.rstrip('\n'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"Selected file set to {selection.file_path}", "preview": preview}

# @router.post("/select")
# async def select_dataset(selection: DatasetSelection):
#     """
#     Select a dataset to be used for training and analysis

#     Args:
#         selection: An object containing the file_path to the selected dataset

#     Returns:
#         Dict with a success message and the selected file
#     """
#     try:
#         file_service.set_selected_dataset(
#             selection.file_path, selection.has_header
#         )

#         preview = []
#         try:
#             with open(selection.file_path, 'r') as f:
#                 for _ in range(10):
#                     line = f.readline()
#                     if not line:
#                         break
#                     preview.append(line.rstrip('\n'))
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Selected file set to {selection.file_path}", "preview": preview}


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

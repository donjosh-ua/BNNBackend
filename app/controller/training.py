import os
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from app.bnn.RNA import train

router = APIRouter()


@router.post("/normal")
async def train_rna():
    """
    Entrena una red neuronal normal.
    """
    try:
        train()
        return {"message": "Entrenamiento de red neuronal normal completado."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

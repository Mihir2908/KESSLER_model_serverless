from fastapi import APIRouter, UploadFile, File
from typing import List
import pandas as pd
from kessler.cdm import ConjunctionDataMessage

router = APIRouter()

@router.post("/cdm/load")
async def load_cdm(file: UploadFile = File(...)):
    # Save uploaded file to a temp location
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    cdm = ConjunctionDataMessage.load(tmp_path)
    return {"cdm": cdm.to_dict()}

@router.post("/cdm/save")
async def save_cdm(cdm_dict: dict):
    cdm = ConjunctionDataMessage()
    for k, v in cdm_dict.items():
        try:
            cdm[k] = v
        except Exception:
            continue
    import tempfile
    file_path = tempfile.mktemp(suffix=".cdm")
    cdm.save(file_path)
    return {"file_path": file_path}

@router.post("/cdm/get_covariance")
async def get_covariance(cdm_dict: dict, object_id: int):
    cdm = ConjunctionDataMessage()
    for k, v in cdm_dict.items():
        try:
            cdm[k] = v
        except Exception:
            continue
    cov = cdm.get_covariance(object_id)
    return {"covariance": cov.tolist()}

@router.post("/cdm/to_dict")
async def cdm_to_dict(cdm_dict: dict):
    cdm = ConjunctionDataMessage()
    for k, v in cdm_dict.items():
        try:
            cdm[k] = v
        except Exception:
            continue
    return {"cdm": cdm.to_dict()}

@router.post("/cdm/to_dataframe")
async def cdm_to_dataframe(cdm_dict: dict):
    cdm = ConjunctionDataMessage()
    for k, v in cdm_dict.items():
        try:
            cdm[k] = v
        except Exception:
            continue
    df = cdm.to_dataframe()
    return {"dataframe": df.to_dict(orient="records")}

@router.post("/cdm/validate")
async def validate_cdm(cdm_dict: dict):
    cdm = ConjunctionDataMessage()
    for k, v in cdm_dict.items():
        try:
            cdm[k] = v
        except Exception:
            continue
    # This prints missing fields to stdout; you may want to capture and return them
    cdm.validate()
    return {"status": "Validation complete"}
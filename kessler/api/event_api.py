from fastapi import APIRouter, UploadFile, File
from typing import List, Optional
import pandas as pd
from kessler.event import EventDataset

router = APIRouter()

@router.post("/event/from_pandas")
async def from_pandas_api(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    event_dataset = EventDataset.from_pandas(df)
    return {"status": "EventDataset created", "num_events": len(event_dataset)}

@router.post("/event/to_dataframe")
async def to_dataframe_api(cdms: List[dict]):
    df = pd.DataFrame(cdms)
    return df.to_dict(orient="records")

@router.post("/event/describe")
async def describe_event(cdms: List[dict]):
    df = pd.DataFrame(cdms)
    return {"description": df.describe().to_dict()}

@router.post("/event/plot_feature")
async def plot_feature_api(cdms: List[dict], feature: str):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(cdms)
    if 'TCA' not in df.columns or feature not in df.columns:
        return {"error": "Required columns not found."}
    df['TCA'] = pd.to_datetime(df['TCA'], errors='coerce')
    plt.figure(figsize=(8, 5))
    plt.plot(df['TCA'], df[feature], marker='o')
    plt.xlabel('TCA')
    plt.ylabel(feature)
    plt.title(f'{feature} vs TCA')
    plt.grid(True)
    plt.tight_layout()
    fname = f'event_{feature.lower()}_vs_tca.pdf'
    plt.savefig(fname)
    plt.close()
    return {"plot": fname}

@router.post("/event/plot_features")
async def plot_features_api(cdms: List[dict], features: Optional[List[str]] = None):
    import matplotlib.pyplot as plt
    import numpy as np
    df = pd.DataFrame(cdms)
    if 'TCA' not in df.columns:
        return {"error": "TCA column not found."}
    df['TCA'] = pd.to_datetime(df['TCA'], errors='coerce')
    df = df.sort_values('TCA')
    plots = []
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in features:
        if col.upper() != 'TCA' and col in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(df['TCA'], df[col], marker='o')
            plt.xlabel('TCA')
            plt.ylabel(col)
            plt.title(f'{col} vs TCA')
            plt.grid(True)
            plt.tight_layout()
            fname = f'event_{col.lower()}_vs_tca.pdf'
            plt.savefig(fname)
            plt.close()
            plots.append(fname)
    return {"plots": plots}

@router.post("/event/plot_uncertainty")
async def plot_uncertainty_api(cdms: List[dict]):
    # Placeholder: implement your uncertainty plotting logic here
    return {"plot": "uncertainty_plot.pdf"}
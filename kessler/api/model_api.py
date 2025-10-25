from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import dsgp4
import pandas as pd
from kessler.model import Conjunction
from kessler import GNSS, Radar

router = APIRouter()

class TLEInput(BaseModel):
    t_tle: List[str]
    c_tle: List[str]

@router.post("/model/forward")
def model_forward(input: TLEInput):
    t_tle = dsgp4.tle.TLE(input.t_tle)
    c_tle = dsgp4.tle.TLE(input.c_tle)
    model = Conjunction(
        t_observing_instruments=[GNSS()],
        c_observing_instruments=[Radar()],
        t_tle=t_tle,
        c_tle=c_tle
    )
    trace, iters = model.get_conjunction()
    cdms = trace.nodes['cdms']['infer']['cdms']
    cdm_dicts = [cdm.to_dict() for cdm in cdms]
    return {"cdms": cdm_dicts, "iterations": iters}

@router.post("/model/plot_cdms")
def plot_cdms(cdms: List[dict], features: Optional[List[str]] = None):
    import matplotlib.pyplot as plt
    import numpy as np
    df = pd.DataFrame(cdms)
    if 'TCA' not in df.columns:
        return {"error": "TCA column not found in CDMs."}
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
            plt.title(f'Synthetic CDMs: {col} vs TCA')
            plt.grid(True)
            plt.tight_layout()
            fname = f'synthetic_cdms_{col.lower()}_vs_tca.pdf'
            plt.savefig(fname)
            plt.close()
            plots.append(fname)
    return {"plots": plots}

@router.post("/model/get_conjunction_summary")
def get_conjunction_summary(cdms: List[dict]):
    df = pd.DataFrame(cdms)
    summary = df.describe().to_dict()
    return {"summary": summary}

@router.post("/model/monte_carlo_propagation")
def monte_carlo_propagation_api(input: TLEInput, num_samples: int = 1000):
    t_tle = dsgp4.tle.TLE(input.t_tle)
    c_tle = dsgp4.tle.TLE(input.c_tle)
    model = Conjunction(
        t_observing_instruments=[GNSS()],
        c_observing_instruments=[Radar()],
        t_tle=t_tle,
        c_tle=c_tle
    )
    # Placeholder: you should call model.propagate_uncertainty_monte_carlo with appropriate arguments
    # result = model.propagate_uncertainty_monte_carlo(...)
    return {"result": "Monte Carlo propagation results (implement logic)"}
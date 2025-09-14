from fastapi import FastAPI
from .event_api import router as event_router
from .model_api import router as model_router
from .nn_api import router as nn_router
from .cdm_api import router as cdm_router

app = FastAPI()

app.include_router(event_router)
app.include_router(model_router)
app.include_router(nn_router)
app.include_router(cdm_router)
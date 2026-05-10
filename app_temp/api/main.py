from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from app.api.core.config import settings
from app.api.routers import monitoring_router, prediction_router


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for Stock Data Prediction using MLflow models and S3",
    # root_path="/prod",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Include Routers
# app.include_router(monitoring_router.router, tags=["monitoring"])
# app.include_router(prediction_router.router, tags=["Prediction"])

# Registrando os roteadores usando o novo caminho
app.include_router(prediction_router.router, prefix="/prod", tags=["Predições"])
app.include_router(monitoring_router.router, prefix="/prod/monitoring", tags=["Observabilidade e ML"])

handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

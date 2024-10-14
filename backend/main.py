# flake8: noqa: E402
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app.settings import init_settings
from app.observability import init_observability
from app.api.routers import api_router
import uvicorn
from fastapi import Request
import os
import logging
from app.config import DATA_DIR
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

init_settings()
init_observability()

# Default to 'development' if not set
environment = os.getenv("ENVIRONMENT", "dev")
logger = logging.getLogger("uvicorn")

# Define allowed origins
ALLOWED_ORIGINS = ["https://ng-cookbook.com"]

if environment == "dev":
    ALLOWED_ORIGINS.extend(["http://localhost:4200", "http://127.0.0.1:4200"])
    logger.warning(
        f"Running in development mode - allowing CORS for: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global OPTIONS handler


@app.options("/{full_path:path}")
async def options_handler(request: Request):
    return JSONResponse(
        status_code=200,
        content={"message": "OK"}
    )

# Redirect to documentation page when accessing base URL


@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


def mount_static_files(directory, path):
    if os.path.exists(directory):
        logger.info(f"Mounting static files '{directory}' at '{path}'")
        app.mount(
            path,
            StaticFiles(directory=directory, check_dir=False),
            name=f"{directory}-static",
        )


# Mount the data files to serve the file viewer
mount_static_files(DATA_DIR, "/api/files/data")
# Mount the output files from tools
mount_static_files("output", "/api/files/output")

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", "8000"))
    reload = True if environment == "dev" else False

    uvicorn.run(app="main:app", host=app_host, port=app_port, reload=reload)

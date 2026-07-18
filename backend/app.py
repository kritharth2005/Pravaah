import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from routers import professional_router
from routers import human_router


load_dotenv()

origins = ["*"]



# Check for API Key
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")

app = FastAPI(
    title="Pravaah Legal AI",
    description="API for summarizing and advising on legal documents.",
    version="1.0.0",
)

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIRECTORY = os.path.join(BASE_DIR, "uploads")
STATIC_DIRECTORY = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pravaah Legal AI API"}


app.include_router(router=human_router.router)
app.include_router(router=professional_router.router)

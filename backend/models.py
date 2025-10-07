from pydantic import BaseModel

class ResponseBody(BaseModel):
    text: str
    audio_path: str

class QueryRequest(BaseModel):
    query: str
    language: str = "en"

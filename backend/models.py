from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from config import MAX_REQUEST_CHARS


class Audience(str, Enum):
    human = "human"
    professional = "professional"


class Mode(str, Enum):
    summary = "summary"
    advice = "advice"


class Language(str, Enum):
    eng = "eng"
    hin = "hin"
    kan = "kan"
    tam = "tam"
    mal = "mal"
    tel = "tel"


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    language: Language = Language.eng


class ResponseBody(BaseModel):
    text: str
    audio_path: str | None = None
    # Non-fatal problems the user should know about (truncated input, audio unavailable, ...).
    notices: list[str] = []

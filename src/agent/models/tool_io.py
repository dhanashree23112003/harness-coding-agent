from pydantic import BaseModel, Field


class ReadFileInput(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file to read")
    encoding: str = Field("utf-8", description="File encoding")


class ReadFileOutput(BaseModel):
    path: str
    content: str
    size_bytes: int

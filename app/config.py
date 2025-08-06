from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    allowed_origins: str = Field(..., description="The allowed origins for cors settings")
    allowed_methods: str = Field(..., description="The allowed methods for cors settings")
    allowed_headers: str = Field(..., description="The allowed headers for cors settings")
    task: str = Field(..., description="The type of inference we are doing")
    model: str = Field(..., description="The model we are using")
    device: str = Field(..., description="The device to use. Uses CUDA if it is available")

app_settings = ApiSettings()
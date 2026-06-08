from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict()

    ENV: str = "local"
    LOGGING_LEVEL: str = Field(default="INFO")

    GRPC_PORT: int = Field(default=50051)
    GRPC_TOKEN: SecretStr = Field(default=...)

    MODEL_PATH: str = Field(default="models/model.onnx")
    ONNX_PROVIDERS: list[str] = Field(default=["CPUExecutionProvider"])

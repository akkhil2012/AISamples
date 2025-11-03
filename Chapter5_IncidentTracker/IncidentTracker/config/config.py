"""
Configuration management for Error Resolution System
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Mistral AI Configuration
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
    mistral_model: str = Field(default="mistral-large-latest", env="MISTRAL_MODEL")
    
    # Microsoft Graph API Configuration
    microsoft_client_id: str = Field(..., env="MICROSOFT_CLIENT_ID")
    microsoft_client_secret: str = Field(..., env="MICROSOFT_CLIENT_SECRET")
    microsoft_tenant_id: str = Field(..., env="MICROSOFT_TENANT_ID")
    
    # Confluence Configuration
    confluence_url: str = Field(..., env="CONFLUENCE_URL")
    confluence_username: str = Field(..., env="CONFLUENCE_USERNAME")
    confluence_api_token: str = Field(..., env="CONFLUENCE_API_TOKEN")
    
    # Application Configuration
    app_host: str = Field(default="localhost", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./error_resolution.db", env="DATABASE_URL")
    
    # MCP Server Ports
    confluence_mcp_port: int = Field(default=8001, env="CONFLUENCE_MCP_PORT")
    teams_mcp_port: int = Field(default=8002, env="TEAMS_MCP_PORT")
    outlook_mcp_port: int = Field(default=8003, env="OUTLOOK_MCP_PORT")
    local_disk_mcp_port: int = Field(default=8004, env="LOCAL_DISK_MCP_PORT")
    
    # Search Configuration
    max_search_results: int = Field(default=50, env="MAX_SEARCH_RESULTS")
    search_timeout_seconds: int = Field(default=30, env="SEARCH_TIMEOUT_SECONDS")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    
    # File Search Configuration
    local_search_paths: str = Field(
        default="/tmp/documents,/tmp/logs", 
        env="LOCAL_SEARCH_PATHS"
    )
    allowed_file_extensions: str = Field(
        default=".txt,.md,.pdf,.docx,.log,.json,.yaml,.yml",
        env="ALLOWED_FILE_EXTENSIONS"
    )
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_burst_size: int = Field(default=10, env="RATE_LIMIT_BURST_SIZE")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    @property
    def search_paths_list(self) -> List[str]:
        """Convert comma-separated search paths to list"""
        return [path.strip() for path in self.local_search_paths.split(',')]
    
    @property
    def file_extensions_list(self) -> List[str]:
        """Convert comma-separated file extensions to list"""
        return [ext.strip() for ext in self.allowed_file_extensions.split(',')]
    
    @property
    def microsoft_authority_url(self) -> str:
        """Generate Microsoft authority URL"""
        return f"https://login.microsoftonline.com/{self.microsoft_tenant_id}"
    
    @property
    def microsoft_graph_scope(self) -> List[str]:
        """Microsoft Graph API scopes"""
        return [
            "https://graph.microsoft.com/Mail.Read",
            "https://graph.microsoft.com/Mail.Send",
            "https://graph.microsoft.com/Chat.Read",
            "https://graph.microsoft.com/Files.Read.All"
        ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


class ErrorSeverity:
    """Error severity levels"""
    P1 = "P1"
    P2 = "P2" 
    P3 = "P3"
    
    @classmethod
    def all(cls):
        return [cls.P1, cls.P2, cls.P3]


class Environment:
    """Environment types"""
    DEV = "Dev"
    TEST = "Test"
    PROD = "Prod"
    
    @classmethod
    def all(cls):
        return [cls.DEV, cls.TEST, cls.PROD]


class SearchSource:
    """Available search sources"""
    CONFLUENCE = "confluence"
    TEAMS = "teams"
    OUTLOOK = "outlook"
    LOCAL_DISK = "local_disk"
    
    @classmethod
    def all(cls):
        return [cls.CONFLUENCE, cls.TEAMS, cls.OUTLOOK, cls.LOCAL_DISK]
    
    @classmethod
    def get_display_name(cls, source: str) -> str:
        display_names = {
            cls.CONFLUENCE: "Confluence Search",
            cls.TEAMS: "Teams Search",
            cls.OUTLOOK: "Outlook Email Search",
            cls.LOCAL_DISK: "Local Disk Search"
        }
        return display_names.get(source, source)


# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.log_level,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "error_resolution.log",
            "mode": "a",
        },
    },
    "root": {
        "level": settings.log_level,
        "handlers": ["console", "file"],
    },
}
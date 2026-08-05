import os
from pathlib import Path

BOX_ROOT = Path.home() / "Library/CloudStorage/Box-Box/160_Construction/02_Realizace/240783160_Garáže_NDS"
APP_SUPPORT_DIR = Path(os.environ.get("AI_SEARCH_HOME",Path.home() / "Library/Application Support/AI Search"))
DATABASE_DIR = APP_SUPPORT_DIR / "database"
LANCE_DIR = APP_SUPPORT_DIR / "lance"
CACHE_DIR = APP_SUPPORT_DIR / "cache"
LOGS_DIR = APP_SUPPORT_DIR / "logs"
STATE_DIR = APP_SUPPORT_DIR / "state"
EMBEDDING_MODEL = "BAAI/bge-m3"
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"
COMPLEX_MODEL = "qwen3:14b"
VISION_MODEL = "gemma4"
PARSE_TIMEOUT_SECONDS = 120
CHUNK_TIMEOUT_SECONDS = 60
EMBEDDING_TIMEOUT_SECONDS = 60
EMBEDDING_BATCH_SIZE = 8
MSG_PARSE_TIMEOUT_SECONDS = 120

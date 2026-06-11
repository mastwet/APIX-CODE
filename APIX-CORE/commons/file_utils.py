from pathlib import Path
import os
import yaml

from .logger import logger


# ==========================================================
# Yaml
# ==========================================================

def load_from_yaml(dir, key=None) -> dict | str:
    """
    Load yaml file and optionally return a specific key.

    Args:
        dir (str): Path to yaml file.
        key (str, optional): Specific key to retrieve from yaml content.
            If provided, return config[key], otherwise return full config.

    Returns:
        dict | str:
            - Full yaml data (dict) if key is None
            - Value of the specified key if key is provided (may be None if key not found)

    Raises:
        Exception: If file reading or yaml parsing fails.
    """
    config = None
    try:
        if os.path.exists(dir):
            with open(dir, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logger.info("[load_from_yaml] load config from yaml file successfully.")
        else:
            config = {}
        if key is not None:
            return config.get(key)
    
    except Exception as e:
        logger.error(f"[load_from_yaml] Error loading yaml file: {e}")
        raise
    return config


def write_to_yaml(dir, data: dict):
    """
    Write data to yaml file (overwrite mode).

    Args:
        dir (str): Path to yaml file.
        data (dict): Data to be written into yaml.

    Returns:
        None

    Raises:
        Exception: If file writing fails.
    """
    try:
        if not os.path.exists(dir):
            Path(dir).parent.mkdir(parents=True, exist_ok=True)
        with open(dir, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)
            logger.info("[write_to_yaml] write data to local yaml file successfully.")
    except Exception as e:
        logger.error(f"[write_to_yaml] Error writing to yaml file: {e}")
        raise


# ==========================================================
# Text Loader
# ==========================================================

def load_text(file_path: str) -> str:
    """
    Load text content from a supported text file.

    Supported formats:
        .txt, .md, .log, .json, .csv,
        .xml, .html, .htm,
        .py, .js, .ts,
        .yaml, .yml

    Returns:
        file content as string

    Raises:
        ValueError: if file extension is not supported
        Exception: if file reading fails
    """

    allowed_types = {
        ".txt",
        ".md",
        ".log",
        ".json",
        ".csv",
        ".xml",
        ".html",
        ".htm",
        ".py",
        ".js",
        ".ts",
        ".yaml",
        ".yml",
    }

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in allowed_types:
        raise ValueError(f"Unsupported text file type: {ext}")

    try:
        # Try reading with utf-8
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except UnicodeDecodeError:
        # Fallback for files with BOM
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return f.read()

    except Exception as e:
        raise e

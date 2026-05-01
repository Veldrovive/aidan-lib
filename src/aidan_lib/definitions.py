from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).parent

data_dir_str = os.getenv("DATA_DIR")
if data_dir_str is None:
    data_dir_str = str(ROOT_DIR.parent.parent / "data")
    print("WARNING: DATA_DIR not found in .env file, defaulting to", data_dir_str)
DATA_DIR = Path(data_dir_str)

from pathlib import Path
from dotenv import load_dotenv
import os
import sys

load_dotenv()

ROOT_DIR = Path(__file__).parent

data_dir_str = os.getenv("DATA_DIR")
if data_dir_str is None:
    data_dir_str = str(ROOT_DIR.parent.parent / "data")
    print("WARNING: DATA_DIR not found in .env file, defaulting to", data_dir_str)
DATA_DIR = Path(data_dir_str)

DINOV3_DIR = ROOT_DIR / "dinov3_repo"
sys.path.append(str(DINOV3_DIR))

DINOV3_VITS16_URL = os.getenv("DINOV3_VITS16_URL")
DINOV3_VITS16_PLUS_URL = os.getenv("DINOV3_VITS16_PLUS_URL")
DINOV3_VITB16_URL = os.getenv("DINOV3_VITB16_URL")
DINOV3_VITL16_URL = os.getenv("DINOV3_VITL16_URL")
DINOV3_VITH16PLUS_URL = os.getenv("DINOV3_VITH16PLUS_URL")
DINOV3_VIT7B16_URL = os.getenv("DINOV3_VIT7B16_URL")

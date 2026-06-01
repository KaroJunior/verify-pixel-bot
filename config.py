import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in environment variables")

# Paths
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"

# Ensure temp directory exists
TEMP_DIR.mkdir(exist_ok=True)

# Analysis weights (sum = 1.0)
WEIGHTS = {
    'metadata': 0.25,   # 25% weight
    'frequency': 0.40,  # 40% weight  
    'heuristic': 0.35   # 35% weight
}

# Analysis thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.7,    # 70%+ confidence
    'medium': 0.4,  # 40-69% confidence
    'low': 0.0      # below 40% confidence
}

# Cleanup settings
MAX_IMAGE_SIZE_MB = 20  # Maximum image size to process
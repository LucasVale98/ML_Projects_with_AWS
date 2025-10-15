import logging
import os
from datetime import datetime

# 1. Define Paths
LOG_FILE = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log" # Added time for unique filenames
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# 2. Define Format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# 3. Create Handlers
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT)) # Set format for file

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(LOG_FORMAT)) # Set format for console

# 4. Configure BasicConfig using ONLY handlers and general settings (level)
logging.basicConfig(
    level=logging.INFO, # General logging level (applies to all handlers by default)
    handlers=[
        file_handler,    # Writes to the file
        stream_handler   # Prints to the console
    ]
)

 
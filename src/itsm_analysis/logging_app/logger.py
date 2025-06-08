import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

path_logs = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(path_logs, exist_ok=True)

PATH_LOG_FILE= os.path.join(path_logs, LOG_FILE)

logging.basicConfig(
    filename=PATH_LOG_FILE,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
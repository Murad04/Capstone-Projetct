from functools import wraps
import logging
from logging.handlers import RotatingFileHandler

import logging
from logging.handlers import RotatingFileHandler

# Custom handler for immediate flushing
class ImmediateFlushRotatingFileHandler(RotatingFileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # Ensure immediate flush after writing

# Configure a rotating file handler with immediate flush
handler = ImmediateFlushRotatingFileHandler(
    r'D:\\Personal\\codes\\project capstone\\cloud\\tmp\\app.log',
    maxBytes=1024 * 1024,  # 1 MB
    backupCount=2          # Keep 2 backup files
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[handler]  # Attach the custom handler
)

def log_function(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logging.info(f'Starting {func.__name__}')
        result = await func(*args, **kwargs)
        logging.info(f'Completed {func.__name__}')
        return result
    return wrapper
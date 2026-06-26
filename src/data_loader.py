import pandas as pd
from src.logs import get_logger  # Importing get_logger() function that created in logs.py

logger = get_logger()  # creates a logger object

def load_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:  # If anything goes wrong inside try block jump to here
        logger.error(f"Error loading data: {e}")
        return None
    

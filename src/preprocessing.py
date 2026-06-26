#It will clean the dataset before training the model.
from src.logs import get_logger #Imports get_logger() from logs.py

logger = get_logger() #creates a loger object

def clean_data(df): # creates a function named as clean_data
    initial_shape = df.shape #before cleaning saving the initial shape to compare before & after cleaning.

    df = df.dropna() # Removes rows containg missing(NaN) values.

    logger.info(f"Cleaned data: {initial_shape} -> {df.shape}") #It contain Intial shape->and the shape after cleaning. 
    return df #returns cleaned value
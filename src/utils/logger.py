import logging
import os

# Define log directory
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Define log file names
LOG_FILE = os.path.join(LOG_DIR, 'project_log.log')

def setup_logger():
    """
    Setup the logger for the project.
    Returns:
        logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger('project_logger')
    logger.setLevel(logging.DEBUG)
    
    # File handler to log messages to a file
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    
    # Stream handler to log messages to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Print only INFO and above to the console
    
    # Log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

# Example of using the logger
def example_usage():
    logger = setup_logger()
    
    try:
        # Simulating a model training process
        logger.info("Starting model training...")
        # Simulated model code
        # model.train()
        logger.info("Model training completed successfully.")
    except Exception as e:
        logger.error(f"Error occurred during model training: {str(e)}", exc_info=True)

if __name__ == "__main__":
    example_usage()

import logging  # Imports built in logging module.

def get_logger():
    logging.basicConfig(
        filename="logs/project.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )  # will setup how loggging should behave
    return logging.getLogger()
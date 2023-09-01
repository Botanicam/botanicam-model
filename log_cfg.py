import logging

# Create a logger and configure it if it hasn't been created yet
if not logging.getLogger().handlers:
    def setup_logging():
        # Create a logger
        logger = logging.getLogger("my_logger")
        logger.setLevel(logging.DEBUG)

        # Create a formatter with the desired log format
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%d/%m/%Y %H:%M"
        )

        # Create a file handler (logs to a file)
        file_handler = logging.FileHandler("my_app.log")
        file_handler.setFormatter(formatter)

        # Create a console handler (logs to the console)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    # Create a logger
    logger = setup_logging()
else:
    # If the logger has already been configured, just get the existing logger
    logger = logging.getLogger("my_logger")
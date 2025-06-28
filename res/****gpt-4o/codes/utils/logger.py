import logging, os


# Define loggers
PRED_LOGGER = logging.getLogger("pred_log")
ERR_LOGGER = logging.getLogger("error_log")


def setup_logger(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    
    formatter = logging.Formatter(
        # fmt='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s',
        fmt='%(asctime)s --->   %(message)s',
        datefmt='%H:%M:%S'
    )

    logger_configs = {
        PRED_LOGGER: os.path.join(log_folder, "pred.log"),
        ERR_LOGGER: os.path.join(log_folder, "error.log"),
    }

    for logger, log_path in logger_configs.items():
        handler = logging.FileHandler(log_path, mode='a')
        handler.setFormatter(formatter)
        logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
        logger.addHandler(handler)
        logger.propagate = False

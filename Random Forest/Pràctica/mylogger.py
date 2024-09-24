import logging


def mylogger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s.py %(lineno)d - %(asctime)s - '
                                  '%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

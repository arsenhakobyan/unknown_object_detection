import os
import sys
import logging
import traceback


def setup_logging(logger, level='debug'):
    logger.setLevel(logging.INFO)
    logging.addLevelName(
        logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(
        logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(
        logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(
        logging.DEBUG, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))

    logformatter = '%(asctime)s [%(levelname)s] [%(module)s] %(filename)s:%(lineno)d %(message)s'
    loglevel = logging.INFO
    logging.basicConfig(format=logformatter, level=loglevel)
    if level == "info":
        logger.setLevel(logging.INFO)
    if level == "debug":
        logger.setLevel(logging.DEBUG)
    if level == "warning":
        logger.setLevel(logging.WARNING)
    if level == "disable":
        logger.disabled = True

def dump_error():
    global logger
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    var = traceback.format_exc()
    logger.error(str(var) + " at " + str(fname) + ":" + str(exc_tb.tb_lineno))

logger = logging.getLogger(__name__)
setup_logging(logger)


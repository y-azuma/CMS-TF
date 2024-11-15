import logging
import os

class Logger(object):
    def __init__(self, log_config):
        self.save_log_path = log_config.save_log_path
        self.save_log_level = getattr(logging,log_config.save_log_level.upper(),logging.WARNING)
        self.set_config()
        
    def set_config(self):
        # ログ設定
        logging.basicConfig(level=self.save_log_level, format="%(asctime)s - %(levelname)s - %(message)s", filename=self.save_log_path, filemode="a")

        # コンソール出力
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    def log(self, log_level:str, message:str):
        if log_level == "debug":
            self._debug(message)
        elif log_level == "info":
            self._info(message)
        elif log_level == "warning":
            self._warning(message)
        elif log_level == "error":
            self._error(message)
        elif log_level == "critical":
            self._critical(message)
        else:
            error_message = "Error:No such log level :" + str(log_level)
            self._error(error_message)
        
    def _debug(self, message):
        logging.debug(message)
        
    def _info(self, message):
        logging.info(message)
        
    def _warning(self, message):
        logging.warning(message)
        
    def _error(self, message):
        logging.error(message)
        
    def _critical(self, message):
        logging.critical(message)

        
        
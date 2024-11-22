import logging

from cms.modules.parameter_manager import LogParam
class Logger(object):
    """
    Custom logger class

    This class provides a simplified interface for logging messages at different
    levels and configures both file and console output.
    """
    def __init__(self, log_config: LogParam):
        """
        Args:
            log_config (LogParam): Configuration parameters for the logger
        """
        self.save_log_path = log_config.save_log_path
        self.save_log_level = getattr(logging,log_config.save_log_level.upper(),logging.WARNING)
        self.set_config()
        
    def set_config(self) -> None:
        """
        Configure the logger with file and console handlers.
        """
        # Basic configuration for logging to a file
        logging.basicConfig(level=self.save_log_level, format="%(asctime)s - %(levelname)s - %(message)s", filename=self.save_log_path, filemode="a")

        # Set up console output handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    def log(self, log_level: str, message: str) -> None:
        """
        Log a message at the specified log level.

        Args:
            log_level (str): Level at which to log the message
            message (str): Message to be logged.
        """
        # Call the appropriate method based on the log level
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
        
    def _debug(self, message: str) -> None:
        """Log a debug message"""
        logging.debug(message)
        
    def _info(self, message: str) -> None:
        """Log an info message"""
        logging.info(message)
        
    def _warning(self, message: str) -> None:
        """Log a warning message"""
        logging.warning(message)
        
    def _error(self, message: str) -> None:
        """Log an error message"""
        logging.error(message)
        
    def _critical(self, message: str) -> None:
        """Log a critical message"""
        logging.critical(message)

        
        
"""
Monitoring and Logging for Production Applications
Structred logging, metrics and alerts
"""

import json
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

load_dotenv()


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON strings.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "function": record.funcName,
            "thread": record.threadName,
            "process": record.processName,
            "args": record.args,
            "exc_info": record.exc_info,
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging() -> logging.Logger:
    """
    Sets up the logging configuration for the application.
    """
    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging setup complete", extra={"extra_data": {"app": "langgraph"}})

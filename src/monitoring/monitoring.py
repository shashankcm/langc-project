"""
Monitoring and Logging for Production Applications
Structred logging, metrics and alerts
"""

import json
import logging
import os
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
os.environ["LANGCHAIN_TRACING_V2"] = "true"


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


class MetricsCollector:
    """
    Collects and stores metrics for the application.
    """

    def __init__(self):
        self.metrics = {
            "request_total": 0,
            "error_total": 0,
            "latency_count": 0,
            "latency_sum": 0.0,
            "token_input": 0,
            "token_output": 0,
            "cache_hit": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool,
        error: bool = False,
    ):
        self.metrics["request_total"] += 1
        self.metrics["latency_count"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["token_input"] += input_tokens
        self.metrics["token_output"] += output_tokens

        if error:
            self.metrics["error_total"] += 1

        if cache_hit:
            self.metrics["cache_hit"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_summary(self) -> dict:
        avg_latency = (
            self.metrics["latency_sum"] / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )

        error_rate = (
            self.metrics["error_total"] / self.metrics["request_total"]
            if self.metrics["request_total"] > 0
            else 0
        )

        cache_hit_rate = (
            self.metrics["cache_hit"]
            / (self.metrics["cache_hit"] + self.metrics["cache_misses"])
            if self.metrics["cache_hit"] + self.metrics["cache_misses"] > 0
            else 0
        )

        return {
            "total_requests": self.metrics["request_total"],
            "total_errors": self.metrics["error_total"],
            "avg_latency_ms": round(avg_latency, 2),
            "total_token_input": self.metrics["token_input"],
            "total_token_output": self.metrics["token_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "error_rate": f"{error_rate:.2%}",
        }


class InstrumentedLLM:
    """LLM wrapper that instruments calls with monitoring metrics."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.metrics = MetricsCollector()
        self.logger = setup_logging()

    @traceable(name="instrumented_llm_invoke")
    def invoke(self, query: str) -> str:
        start_time = time.time()
        error = False

        try:
            response = self.llm.invoke(query)
            result = response.content

            input_tokens = len(query.split()) * 4 // 3
            output_tokens = len(result.split()) * 4 // 3

            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit=False,
                error=False,
            )

            self.logger.info(
                "LLM Request processed",
                extra={
                    "extra_data": {
                        "latency_ms": (time.time() - start_time) * 1000,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_hit": False,
                        "error": False,
                    }
                },
            )
            return result
        except Exception as e:
            error = True
            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=0,
                output_tokens=0,
                cache_hit=False,
                error=error,
            )
            self.logger.error(
                f"LLM Request failed: {e}", extra={"extra_data": {"error": str(e)}}
            )
            raise e


def demo_monitoring() -> None:
    """Demonstrates the monitoring setup by logging and recording requests."""
    llm = InstrumentedLLM()
    print("Monitoring demo started.\n")

    queries = ["What is Python?", "Explain machine learning"]

    for query in queries:
        result = llm.invoke(query)
        print(f"Query: {query[:30]}...\nResult: {result[:30]}...\n")

    print("Monitoring demo completed.\n")

    print("=" * 30)
    print("\nMetrics Summary:\n")
    summary = llm.metrics.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 30)


if __name__ == "__main__":
    # logger = setup_logging()
    # logger.info("Logging setup complete", extra={"extra_data": {"app": "langgraph"}})

    demo_monitoring()

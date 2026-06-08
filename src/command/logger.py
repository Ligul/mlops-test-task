import inspect
import logging
import sys
import traceback
import typing

import orjson
from loguru import logger

if typing.TYPE_CHECKING:
    from loguru import Record


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def serialize(record: "Record") -> str:
    exception = record["exception"]

    if exception is not None:
        exception = {
            "type": None if exception.type is None else exception.type.__name__,
            "value": exception.value,
            "traceback": "".join(traceback.format_tb(exception.traceback)),
        }

    return orjson.dumps(
        {
            "time": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "file": {
                "path": record["file"].path,
                "name": record["file"].name,
            },
            "module": record["module"],
            "name": record["name"],
            "function": record["function"],
            "line": record["line"],
            "extra": record["extra"],
            "exception": exception,
            "thread": record["thread"].name,
        },
        default=str,
    ).decode("utf-8")


def formatter(record: "Record") -> str:
    # Note this function returns the string to be formatted, not the actual message to be logged
    record["extra"]["serialized"] = serialize(record)
    return "{extra[serialized]}\n"


def config_logger(level: str, env: str) -> None:
    # https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.remove()
    if env in ("dev", "test", "local"):
        logger.add(sys.stderr, level=level)
    else:
        logger.add(sys.stderr, level=level, format=formatter)

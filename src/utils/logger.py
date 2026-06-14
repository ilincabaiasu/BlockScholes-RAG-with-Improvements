import json
import logging
import datetime

# All attributes that exist on a standard LogRecord — extras are anything else.
_STANDARD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS
        }
        if extra:
            payload["extra"] = extra

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that emits one JSON line per event.

    Usage::

        logger = get_logger(__name__)
        logger.info("loaded", extra={"count": 42})
    """
    logger = logging.getLogger(name)

    if not logging.root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)

    return logger

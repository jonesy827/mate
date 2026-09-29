"""Bounded per-call traces, without credentials or pre-authentication speech."""

import json
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger("mate.audit")


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if not isinstance(value, str):
        return value
    for key, secret in os.environ.items():
        if secret and len(secret) >= 4 and any(x in key for x in ("SECRET", "API_KEY", "PASSPHRASE")):
            pattern = r"[\W_]+".join(re.escape(w) for w in secret.split())
            value = re.sub(pattern, "[redacted]", value, flags=re.IGNORECASE)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", value)
    return value[:4000]


def event(kind, **fields):
    logger.info(json.dumps({"event": kind, **safe(fields)}, ensure_ascii=True))


class HideSDKTranscripts(logging.Filter):
    def filter(self, record):
        # The SDK logs STT before our authentication gate sees it. Our own
        # authenticated-turn events replace those raw transcript logs.
        return record.getMessage() not in ("received user transcript", "conversation_item_added")


def configure(job_id):
    folder = Path(__file__).resolve().parents[2] / ".logs"
    folder.mkdir(mode=0o700, exist_ok=True)
    path = folder / (re.sub(r"[^a-zA-Z0-9_-]", "_", job_id) + ".log")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    os.close(fd)
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.getLogger("livekit.agents").addFilter(HideSDKTranscripts())
    event("call.started", job_id=job_id)
    return path

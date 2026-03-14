import logging
import os
import sys
from datetime import datetime


def setup_logger(
    name="DynamicEarlyExit",
    log_dir="Logs",
    level=logging.INFO,
):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # IMPORTANT: avoid duplicate logs

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ---- File handler ----
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # ---- Console handler (SLURM captures this) ----
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # Avoid adding handlers twice
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger
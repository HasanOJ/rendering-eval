"""Presentation-only logging setup, outside the inference-source fingerprint."""

import logging


INT8_CAST_MESSAGE = (
    "MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)
_EXPECTED_INT8_CAST_MESSAGES = {
    INT8_CAST_MESSAGE,
    INT8_CAST_MESSAGE.replace("torch.bfloat16", "torch.float32"),
}
_LOGGER_NAME = "bitsandbytes.autograd._functions"


class _ExpectedInt8CastFilter(logging.Filter):
    def filter(self, record):
        return not (
            record.name == _LOGGER_NAME
            and record.levelno == logging.WARNING
            and record.getMessage() in _EXPECTED_INT8_CAST_MESSAGES
        )


def silence_expected_int8_cast_warning():
    """Hide expected BF16/FP32-to-FP16 INT8 warnings; preserve all other logs."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(isinstance(item, _ExpectedInt8CastFilter) for item in logger.filters):
        logger.addFilter(_ExpectedInt8CastFilter())


def main(argv=None):
    """Set up console logging before entering the unchanged evaluation CLI."""
    silence_expected_int8_cast_warning()
    from ..cli import main as cli_main

    return cli_main(argv)

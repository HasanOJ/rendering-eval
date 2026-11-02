import logging

import pytest

from rendering_eval.diagnostics import (
    INT8_CAST_MESSAGE, main, silence_expected_int8_cast_warning,
)


@pytest.fixture
def bnb_logger():
    logger = logging.getLogger("bitsandbytes.autograd._functions")
    previous = list(logger.filters)
    try:
        yield logger
    finally:
        logger.filters[:] = previous


def test_only_expected_int8_cast_warning_is_hidden(caplog, bnb_logger):
    silence_expected_int8_cast_warning()
    count = len(bnb_logger.filters)
    silence_expected_int8_cast_warning()
    assert len(bnb_logger.filters) == count
    with caplog.at_level(logging.WARNING):
        for dtype in ("torch.bfloat16", "torch.float32"):
            bnb_logger.warning(INT8_CAST_MESSAGE.replace("torch.bfloat16", dtype))
            bnb_logger.warning("MatMul8bitLt: inputs will be cast from %s to float16 during quantization",
                               dtype)
        bnb_logger.warning(INT8_CAST_MESSAGE.replace("bfloat16", "float64"))
        bnb_logger.warning("Another bitsandbytes warning")
        bnb_logger.error(INT8_CAST_MESSAGE)
        logging.getLogger("another.backend").warning(INT8_CAST_MESSAGE)
    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        (bnb_logger.name, logging.WARNING, INT8_CAST_MESSAGE.replace("bfloat16", "float64")),
        (bnb_logger.name, logging.WARNING, "Another bitsandbytes warning"),
        (bnb_logger.name, logging.ERROR, INT8_CAST_MESSAGE),
        ("another.backend", logging.WARNING, INT8_CAST_MESSAGE),
    ]


def test_console_setup_filters_before_cli(monkeypatch, caplog, bnb_logger):
    def cli(argv):
        bnb_logger.warning(INT8_CAST_MESSAGE)
        bnb_logger.warning("Unrelated warning")
        return argv
    monkeypatch.setattr("rendering_eval.cli.main", cli)
    with caplog.at_level(logging.WARNING):
        assert main(["run", "--config", "configs/full.yaml"]) == [
            "run", "--config", "configs/full.yaml",
        ]
    assert caplog.messages == ["Unrelated warning"]

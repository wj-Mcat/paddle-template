from __future__ import annotations

from paddle_template.config import (
    Config,
    MetricReport,
    ModelConfigMixin,
    TrainConfigMixin,
)


def test_train_config():
    config = TrainConfigMixin().parse_args(known_only=True)
    assert config.batch_size == 32


def test_model_config():
    config = ModelConfigMixin().parse_args(known_only=True)
    assert config.pretrained_model_or_path == "ernie-1.0"


def test_config():
    config = Config().parse_args(known_only=True)
    assert config.batch_size == 32


def test_metric_report():
    truth, predict = [1, 0], [1, 1]
    report = MetricReport.from_sequence(truth, predict)
    assert report.acc == 0.5

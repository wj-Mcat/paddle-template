from src.config import (
    MetricReport, 
    TrainConfigMixin,
    PredictConfigMixin,
    ModelConfigMixin,
    Config
)


def test_train_config():
    config = TrainConfigMixin().parse_args(known_only=True)
    assert config.batch_size == 32 


def test_predict_config():
    config = PredictConfigMixin().parse_args(known_only=True)
    assert config.model_path is None

def test_model_config():
    config = ModelConfigMixin().parse_args(known_only=True)
    assert config.pretrained_model == 'ernie-1.0'

def test_config():
    config = Config().parse_args(known_only=True)
    assert config.batch_size == 32

def test_metric_report():
    truth, predict = [1, 0], [1, 1]
    report = MetricReport.from_sequence(truth, predict)
    assert report.acc == 0.5

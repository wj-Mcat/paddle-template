from __future__ import annotations
from typing import List, Dict

import paddle
from paddle import nn
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddlenlp.transformers import PretrainedTokenizer, BertTokenizer
from paddle.optimizer.lr import NoamDecay, CosineAnnealingDecay
from paddlespeech.cls.models import cnn14
from paddleaudio.features import LogMelSpectrogram
from paddleaudio.datasets import ESC50
from paddlenlp.utils.log import logger
from tqdm import tqdm
from paddle_template.data_processors import AudioDataProcessor
from paddle_template.config import Config
from paddle_template.trainer import Trainer
from paddle_template.schema import AudioInputExample
from paddle_template.model import SoundClassifier


class SpeechConfig(Config):
    sr: int = 32000
    n_fft = 1024
    win_length = 1024
    hop_length = 320
    f_min: int =50.0
    f_max: int =14000.0
    n_mels: int =64

class TrainerSpeech(Trainer):
    def __init__(self, config: SpeechConfig, processor: AudioDataProcessor, feature_extractor, tokenizer: PretrainedTokenizer, model, criterion: Layer) -> None:
        lr_scheduler = CosineAnnealingDecay(config.learning_rate, T_max=10, last_epoch=100)
        super().__init__(config, processor, tokenizer, model, criterion, lr_scheduler=lr_scheduler)
        self.feature_extractor = feature_extractor

    def create_dataloader(self, dataset):
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=self.config.batch_size)

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            return_list=True
        )
    
    def train_epoch(self):
        self.model.train()
        logger.info(f'training epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))
        for batch_index, features in enumerate(self.train_dataloader):
            waveforms, labels = features
            feats = self.feature_extractor(waveforms)
            feats = paddle.transpose(feats, [0, 2, 1])  # [B, N, T] -> [B, T, N]
            self.on_batch_start()

            # pylint: disable=E1129
            with paddle.amp.auto_cast(
                    self.config.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = self.model(feats)

                loss = self.criterion(logits, labels)
                self.context_data.logits = logits
                self.context_data.loss = loss
                self.context_data.labels = labels

            loss.backward()
            self.on_batch_end()
    
    def evaluate(self, dataloader, mode: str = 'dev'):
        self.model.eval()
        logger.info(f'evaluation epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(dataloader))
        for batch_index, features in enumerate(dataloader):
            waveforms, labels = features
            feats = self.feature_extractor(waveforms)
            feats = paddle.transpose(feats, [0, 2, 1])  # [B, N, T] -> [B, T, N]
            self.on_batch_start()

            # pylint: disable=E1129
            with paddle.amp.auto_cast(
                    self.config.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = self.model(feats)

                loss = self.criterion(logits, labels)
                self.context_data.logits = logits
                self.context_data.loss = loss
                self.context_data.labels = labels

            loss.backward()
            self.on_batch_end()

def main():
    config: SpeechConfig = SpeechConfig().parse_args(known_only=True)
    config.learning_rate = 5e-2
    # 1. create feature extractor
    feature_extractor = LogMelSpectrogram(
        sr=config.sr, 
        n_fft=config.n_fft, 
        hop_length=config.hop_length, 
        win_length=config.win_length, 
        window='hann', 
        f_min=config.f_min,
        f_max=config.f_max,
        n_mels=config.n_mels)
    
    # 2. create model

    model = SoundClassifier(
        cnn14(pretrained=True, extract_embedding=True),
        num_class=len(ESC50.label_list)
    )

    # 3. create dataloader
    criterion = paddle.nn.loss.CrossEntropyLoss()   

    trainer = TrainerSpeech(
        config=config,
        processor=AudioDataProcessor(config.sr),
        tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
        feature_extractor=feature_extractor,
        model=model,
        criterion=criterion,

    )

    trainer.train()


main()
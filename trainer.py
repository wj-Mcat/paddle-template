"""
Authors:    Jingjing WU (吴京京) <https://github.com/wj-Mcat>


2022-now @ Copyright wj-Mcat

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List
from numpy import place

import paddle
from loguru import logger
from paddle.io import DataLoader
from paddle.metric.metrics import Metric
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from tqdm import tqdm
from visualdl import LogWriter

from src.data_processors import DataProcessor
from src.config import get_logger, Config, Tensor
from src.schema import InputExample, ExampleDataset
from src.utils import num


logger = get_logger()


class ContextContainer:
    """Context data container for training
    """
    def __init__(self) -> None:
        """init the variables"""
        self.train_step: int = 0
        self.dev_step: int = 0
        self.epoch: int = 0

        self.train_acc: float = 0
        self.dev_acc: float = 0

        self.loss = 0
        self.dev_loss = 0
        self.logits = 0
        self.labels = 0

        self._cache = defaultdict(int)

    def __getitem__(self, key):
        return self._cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value


class Trainer:
    """Trainer which can handle the tarin/eval/test/predict stage of the model
    """
    def __init__(
        self, config: Config,
        processor: DataProcessor, tokenizer: PretrainedTokenizer,
        model, criterion: Layer,
        optimizer: Optimizer
    ) -> None:
        self.config: Config = config
        self.set_device()

        # 2. build data related objects
        self.data_processor = processor
        self.train_dataloader = self.create_dataloader(processor.get_train_examples())
        self.dev_dataloader = self.create_dataloader(processor.get_dev_examples())
        self.test_dataloader = self.create_dataloader(processor.get_test_examples())

        # 3. init model related
        self.model = model 

        self.train_bar = None

        self.tokenizer = tokenizer

        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        self.optimizer: Optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=config.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)

        self.criterion = criterion

        self.metric: Metric = paddle.metric.Accuracy()

        self.context_data = ContextContainer()
        self._init_output_dir()
        self.writer: LogWriter = LogWriter(logdir=config.output_dir)

    def create_dataloader(self, examples: List[InputExample]):
        dataset = ExampleDataset(examples)
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=self.config.batch_size)

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=lambda _examples: self.convert_examples_to_features(_examples, dataset.label2idx),
            return_list=True
        )
    
    def convert_examples_to_features(self, examples: List[InputExample], label2idx: Dict[str, int]):
        texts = []
        for example in examples:
            if example.text_b is not None:
                texts.append((example.text_a, example.text_b))
            else:
                texts.append(example.text_a)

        # TODO: many-class classification is not supported
        labels = paddle.to_tensor([label2idx[example.label] for example in examples], place=self.config.place())
        encoded_features = self.tokenizer.batch_encode(
            texts,
            max_seq_len=self.config.max_seq_length,
            pad_to_max_seq_len=True,
            return_token_type_ids=True,
        )
        encoded_features.to(self.config.place())
        encoded_features['labels'] = labels
        return encoded_features
        
    def _init_output_dir(self):
        if os.path.exists(self.config.output_dir):
            shutil.rmtree(self.config.output_dir)
        os.makedirs(self.config.output_dir)

    def set_device(self):
        """set paddle device
        """
        paddle.set_device(self.config.device)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.init_parallel_env()

    @paddle.no_grad()
    def evaluate(self, dataloader: DataLoader, mode: str = 'dev'):
        """handle the evaluation based on dataloader

        Args:
            dataloader (DataLoader): the source of dataloader
            mode (str, optional): dev/test. Defaults to 'dev'.
        """
        logger.success(f'{mode} stage ...')
        self.model.eval()
        self.metric.reset()

        # 1. predict the labels based on dataloader
        all_loss = 0
        predicted, truth = [], []

        progress_bar = tqdm(total=len(dataloader))
        for features in dataloader:
            
            labels = features.pop('labels')

            # pylint: disable=E1129
            with paddle.amp.auto_cast(
                self.config.use_amp,
                custom_white_list=["layer_norm", "softmax", "gelu"]
            ):
                logits: Tensor = self.model(**features)
                loss = self.criterion(logits, labels)

            # Get max probs label's index
            pred_index = logits.argmax(axis=-1).detach().numpy().tolist()
            predicted.extend(pred_index)

            labels = labels.detach().numpy().tolist()
            truth.extend(labels)

            sub_acc = sum([pred_index[index] == labels[index]
                          for index in range(len(labels))]) / len(labels)

            progress_bar.update()
            progress_bar.set_description(
                f'loss: {loss:10.4f} acc: {sub_acc: 10.4f}'
            )

        # 2. compute the metric
        assert len(predicted) == len(truth)
        acc = sum([predicted[index] == truth for index in range(
            len(predicted))]) / len(predicted)
        self.context_data.dev_acc = acc
        self.context_data.dev_loss = all_loss

        logger.info(f"eval accuracy: {acc:10.4f} loss: {all_loss:10.4f}")

        self.model.train()
        self.metric.reset()

        self.context_data.dev_step += 1
        self.writer.add_scalar(tag='eval-acc', value=acc,
                               step=self.context_data.dev_step)

        if acc > self.context_data.dev_step:
            self.context_data.dev_acc = acc
            logger.success('saving the best model ...')
            best_model_file = os.path.join(
                self.config.output_dir, 'best.pdparams')
            paddle.save(self.model.state_dict(), best_model_file)

    def _update_bar_info(self):
        bar_info = []
        bar_info.append(f'train-loss: {num(self.context_data.loss):10.6f}')
        bar_info.append(f'train-acc: {self.context_data.train_acc:10.6f}')
        bar_info.append(f'dev-acc: {self.context_data.dev_acc:10.6f}')

        self.train_bar.set_description('\t'.join(bar_info))

    def on_batch_end(self):
        """handle the on batch training is ending logits
        """
        # 1. update global step
        self.context_data.train_step += 1
        self.train_bar.update()

        # 2. compute acc on training dataset
        self.writer.add_scalar(
            'train-loss',
            step=self.context_data.train_step,
            value=self.context_data.loss
        )
        self.writer.add_scalar(
            'train-acc',
            step=self.context_data.train_step,
            value=self.context_data.train_acc
        )
        self._update_bar_info()

        # 3. step the grad
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.clear_grad()

        # 4. eval on dev dataset
        if self.context_data.train_step % self.config.valid_steps == 0:
            self.evaluate(self.dev_dataloader)
            logger.info(
                'saving the model state dict in step: '
                f'{self.context_data.train_step} ...'
            )
            last_model_file = os.path.join(
                self.config.output_dir, 'last.pdparams')
            paddle.save(self.model.state_dict(), last_model_file)
        self.context_data.epoch += 1

    def on_batch_start(self):
        """handle the logit of batch start"""
        self.metric.reset()

    def train_epoch(self):
        """handle the logit of training epoch

        Args:
            epoch (int): _description_
        """
        self.model.train()
        logger.info(f'training epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))

        for features in self.train_dataloader:
            labels = features.pop('labels')
            self.on_batch_start()

            # pylint: disable=E1129
            with paddle.amp.auto_cast(
                    self.config.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = self.model(**features)

                loss = self.criterion(logits, labels)

                self.context_data.logits = logits
                self.context_data.loss = loss
                self.context_data.labels = labels

            loss.backward()
            self.on_batch_end()

    def train(self):
        """the main train epoch"""
        for _ in range(self.config.epochs):
            if self.config.do_train:
                self.train_epoch()

            if self.config.do_dev:
                self.evaluate(self.test_dataloader, mode='test')

    def predict(self, example: InputExample):
        """predict the example"""


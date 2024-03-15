import torch
import numpy as np
from pathlib import PurePath
from ogb.graphproppred import Evaluator

from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy

from code.LMs.lm_model import BertClassifier, BertClaInfModel, BertRegressor, BertRegInfModel
from code.data_utils.dataset import DatasetLoader, LMDataset
from code.utils import time_logger, init_path, project_root_path


def reg_compute_metrics(pred):
    from sklearn.metrics import mean_squared_error
    predictions, labels = pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


def cls_compute_metrics(pred):
    from sklearn.metrics import accuracy_score
    pred, labels = pred
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, cfg):
        self.dataset = cfg.dataset
        self.text = cfg.data.text
        self.seed = cfg.seed

        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink
        self._get_evaluator()

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        self.output_dir = PurePath(
            project_root_path, "output", "lms", self.dataset, self.text,
            "{}-seed{}".format(self.model_name, self.seed)
        )
        self.ckpt_dir = PurePath(
            project_root_path, "output", "prt_lms", self.dataset, self.text,
            "{}-seed{}".format(self.model_name, self.seed)
        )
        # print(self.output_dir)
        # print(self.ckpt_dir)

        # Preprocess data
        self.g_dataset, self.lm_dataset = self.preprocess_data()
        self.eval_metric = self.g_dataset.eval_metric
        self.task_type = self.g_dataset.task_type
        self.num_graphs = self.g_dataset.y.size(0)
        self.n_labels = self.g_dataset.num_classes if 'classification' in self.task_type else 1
        self.train_mask = self.g_dataset.get_idx_split()['train'],
        self.val_mask = self.g_dataset.get_idx_split()['valid'],
        self.test_mask = self.g_dataset.get_idx_split()['test']
        self.compute_metrics = cls_compute_metrics if 'classification' in self.task_type \
            else reg_compute_metrics

        self.inf_dataset = self.lm_dataset
        self.train_dataset = torch.utils.data.Subset(
            self.lm_dataset, self.g_dataset.get_idx_split()['train']
        )
        self.val_dataset = torch.utils.data.Subset(
            self.lm_dataset, self.g_dataset.get_idx_split()['valid']
        )
        self.test_dataset = torch.utils.data.Subset(
            self.lm_dataset, self.g_dataset.get_idx_split()['test']
        )

        self.model = self.setup_model()
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Num. of parameters: {}".format(trainable_params))
        # print(summary(model=model, list(train_loader)[0].x, edge_index))

    def preprocess_data(self):
        # Preprocess data
        dataloader = DatasetLoader(name=self.dataset, text=self.text)
        g_dataset, text = dataloader.dataset, dataloader.text

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        X = tokenizer(text, padding=True, truncation=True, max_length=512)

        lm_dataset = LMDataset(X, g_dataset.y.tolist())

        return g_dataset, lm_dataset

    def setup_model(self):
        # Define pretrained tokenizer and model
        bert_model = AutoModel.from_pretrained(self.model_name)

        if 'classification' in self.task_type:
            model = BertClassifier(
                bert_model,
                n_labels=self.n_labels,
                feat_shrink=self.feat_shrink
            )
        else:
            model = BertRegressor(
                bert_model,
                feat_shrink=self.feat_shrink
            )
        model.config.dropout = self.dropout
        model.config.attention_dropout = self.att_dropout

        return model

    @time_logger
    def train(self):
        # Define training parameters
        eq_batch_size = self.batch_size * 4
        train_steps = self.num_graphs // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        train_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=True,
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )

        # Train pre-trained model
        trainer.train()
        # torch.save(self.model.state_dict(), init_path("{}.ckpt".format(self.ckpt_dir)))
        # print('LM saved to {}.ckpt'.format(self.ckpt_dir))

    def _get_evaluator(self):
        self.evaluator = Evaluator(name=self.dataset)

    @time_logger
    @torch.no_grad()
    def eval(self):
        emb = np.memmap(
            init_path("{}.emb".format(self.ckpt_dir)),
            dtype=np.float16,
            mode='w+',
            shape=(self.num_graphs, self.feat_shrink if self.feat_shrink else 768)
        )
        pred = np.memmap(
            init_path("{}.pred".format(self.ckpt_dir)),
            dtype=np.float16,
            mode='w+',
            shape=(self.num_graphs, self.n_labels)
        )

        if 'classification' in self.task_type:
            inf_model = BertClaInfModel(
                self.model, emb, pred, feat_shrink=self.feat_shrink
            )
        else:
            inf_model = BertRegInfModel(
                self.model, emb, pred, feat_shrink=self.feat_shrink
            )
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*8,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.lm_dataset)

        def _eval(index):
            y_true = torch.Tensor(self.lm_dataset.labels)[index].view(-1, 1)
            y_pred = torch.Tensor(np.argmax(pred[index], -1)).view(-1, 1)

            input_dict = {"y_true": y_true, "y_pred": y_pred}

            return self.evaluator.eval(input_dict)[self.eval_metric]

        train_res = _eval(self.train_mask)
        val_res = _eval(self.val_mask)
        test_res = _eval(self.test_mask)
        print(
            '[LM] Train-{}: {:.4f}, Val-{}: {:.4f}, Test-{}: {:.4f}\n'.format(
                self.eval_metric, train_res,
                self.eval_metric, val_res,
                self.eval_metric, test_res
            )
        )

        return {'TrainRes': train_res, 'ValRes': val_res, 'TestRes': test_res}

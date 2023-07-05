import logging
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from deepmath.deephol.train import torch_training_module
from deepmath.deephol.train.sequence_datamodule import HOListSequenceModule
from deepmath.models.get_model import get_model
from deepmath.models.tactic_predictor import TacticPrecdictor, CombinerNetwork


class ExperimentRunner:
    def __init__(self, exp_config, embedding_model_config):
        self.data_module = HOListSequenceModule(dir=exp_config['data_dir'], batch_size=exp_config['batch_size'])

        self.experiment = torch_training_module.HOListTraining_(embedding_model_goal=get_model(embedding_model_config),
                                                                embedding_model_premise=get_model(embedding_model_config),
                                                                tac_model=TacticPrecdictor(
                                                                    embedding_dim=exp_config['final_embed_dim'],
                                                                    num_tactics=exp_config['num_tactics']),
                                                                combiner_model=CombinerNetwork(
                                                                    embedding_dim=exp_config['final_embed_dim'],
                                                                    num_tactics=exp_config['num_tactics'],
                                                                    tac_embed_dim=exp_config['tac_embed_dim']),
                                                                lr=exp_config['learning_rate'],
                                                                batch_size=exp_config['batch_size'])

        self.logger = WandbLogger(project=exp_config['project'],
                                  name=exp_config['name'],
                                  config={'model_config': embedding_model_config, 'exp_config': exp_config},
                                  )

        self.callbacks = []

        checkpoint_callback = ModelCheckpoint(monitor="rel_param_acc", mode="max",
                                              auto_insert_metric_name=True,
                                              save_top_k=3,
                                              filename="{epoch}-{rel_param_acc}-{topk_acc}",
                                              save_on_train_epoch_end=True,
                                              save_last=True,
                                              save_weights_only=True,
                                              dirpath=exp_config['checkpoint_dir'])

        self.callbacks.append(checkpoint_callback)

        self.trainer = pl.Trainer(enable_progress_bar=True,
                                  max_epochs=exp_config['epochs'],
                                  devices=exp_config['device'],
                                  val_check_interval=exp_config['val_frequency'],
                                  limit_val_batches=exp_config['val_size'],
                                  logger=self.logger,
                                  callbacks=self.callbacks)

    def run(self):
        self.trainer.fit(model=self.experiment, datamodule=self.data_module)

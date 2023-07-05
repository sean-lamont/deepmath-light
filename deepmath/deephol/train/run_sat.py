import logging
from lightning.pytorch.callbacks import ModelCheckpoint

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from deepmath.deephol.train import torch_training_module
from deepmath.deephol.train.torch_data_module import HOListGraphModule
from deepmath.models.get_model import get_model
from deepmath.models.tactic_predictor import TacticPrecdictor, CombinerNetwork

if __name__ == "__main__":
    NUM_TOKENS = 2044 + 5
    logging.basicConfig(level=logging.DEBUG)

    exp_config = {
        "learning_rate": 1e-4,
        "epochs": 20,
        "weight_decay": 1e-6,
        "batch_size": 16,
        "model_save": False,
        "val_size": 512,
        "logging": False,
        "checkpoint_dir": "/home/sean/Documents/phd/deepmath-light/deepmath/train/",
        "device": [0],
        "val_frequency": 2048,
        "num_tactics": 41,
        "tac_embed_dim": 128,
        "final_embed_dim": 1024
    }

    model_config = {
        "model_type": "sat",
        # 'gnn_type': 'di_gcn',
        "num_edge_features": 3,
        "vocab_size": NUM_TOKENS,
        "embedding_dim": 256,
        "dim_feedforward": 256,
        "num_heads": 4,
        "num_layers": 4,
        "in_embed": True,
        "se": "gnn-encoder",
        "abs_pe": False,
        "abs_pe_dim": 256,
        "use_edge_attr": True,
        "dropout": 0.5,
        "gnn_layers": 4,
        'small_inner': True
    }

    data_config = {"source": "graph", "dir": '/home/sean/Documents/phd/deepmath-light/deepmath/attention_train_data/',
                   'batch_size': exp_config['batch_size']}

    data_module = HOListGraphModule(config=data_config)

    experiment = torch_training_module.HOListTraining_(embedding_model_goal=get_model(model_config),
                                                       embedding_model_premise=get_model(model_config),
                                                       tac_model=TacticPrecdictor(
                                                           embedding_dim=exp_config['final_embed_dim'],
                                                           num_tactics=exp_config['num_tactics']),
                                                       combiner_model=CombinerNetwork(
                                                           embedding_dim=exp_config['final_embed_dim'],
                                                           num_tactics=exp_config['num_tactics'],
                                                           tac_embed_dim=exp_config['tac_embed_dim']),
                                                       lr=exp_config['learning_rate'],
                                                       batch_size=exp_config['batch_size'])

    logger = WandbLogger(project='HOList Pretrain',
                         name='SAT Directed Attention No PE Large',
                         config={'model_config': model_config, 'exp_config': exp_config},
                         # offline=True,
                         )

    callbacks = []

    # todo update model artifacts manually
    checkpoint_callback = ModelCheckpoint(monitor="rel_param_acc", mode="max",
                                          auto_insert_metric_name=True,
                                          save_top_k=3,
                                          filename="{epoch}-{rel_param_acc}-{topk_acc}",
                                          save_on_train_epoch_end=True,
                                          save_last=True,
                                          save_weights_only=True,
                                          dirpath=exp_config['checkpoint_dir'])

    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(enable_progress_bar=True,
                         max_epochs=exp_config['epochs'],
                         devices=exp_config['device'],
                         val_check_interval=exp_config['val_frequency'],
                         limit_val_batches=exp_config['val_size'],
                         logger=logger,
                         callbacks=callbacks)

    trainer.fit(model=experiment, datamodule=data_module)



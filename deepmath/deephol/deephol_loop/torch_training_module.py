import traceback
import wandb
import warnings
import einops

warnings.filterwarnings('ignore')
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
# from models.get_model import get_model
# from models.gnn.formula_net.formula_net import BinaryClassifier
from deepmath.deephol.deephol_loop import torch_data_module
import torch

def auroc(pos, neg):
    return torch.sum(torch.log(1 + torch.exp(-1 * (pos - neg))))


def binary_loss(preds, targets):
    return -1. * torch.sum(targets * torch.log(preds) + (1 - targets) * torch.log((1. - preds)))


'''

Calculate loss function as defined in original implementation paper. 

neg_premise_scores are scores for negative premises, and extra_neg_premises are additional negatives sampled from the batch.
They are weighted differently to favour negatives from the same goal

'''


class HOListTraining(pl.LightningModule):
    def __init__(self,
                 embedding_model_goal,
                 embedding_model_premise,
                 tac_model,
                 combiner_model,
                 batch_size=32,
                 lr=1e-4):
        super().__init__()

        self.embedding_model_goal = embedding_model_goal
        self.embedding_model_premise = embedding_model_premise
        self.tac_model = tac_model
        self.tac_model = tac_model
        self.combiner_model = combiner_model
        self.eps = 1e-6
        self.lr = lr
        self.batch_size = batch_size

        self.save_hyperparameters()

    def loss_func(self, tac_pred, true_tac, pos_premise_scores, neg_premise_scores, extra_neg_premise_scores,
                  tac_weight=1,
                  pairwise_weight=0.2,
                  auroc_weight=4,
                  same_goal_weight=2):

        # bce for tac
        tac_loss = binary_loss(tac_pred, true_tac)
        pairwise_loss_positives = binary_loss(pos_premise_scores, torch.ones(pos_premise_scores.shape[0]))

        pairwise_loss_main_negatives = binary_loss(neg_premise_scores, torch.zeros(neg_premise_scores.shape[0]))
        pairwise_loss_extra_negatives = binary_loss(extra_neg_premise_scores,
                                                    torch.zeros(extra_neg_premise_scores.shape[0]))

        pos_premise_scores_main_negatives = einops.repeat(pos_premise_scores, 'b 1 -> b k',
                                                          k=neg_premise_scores.shape[-1])
        auroc_loss_main_negatives = auroc(pos_premise_scores_main_negatives, neg_premise_scores)

        pos_premise_scores_extra_negatives = einops.repeat(pos_premise_scores, 'b 1 -> b k',
                                                           k=extra_neg_premise_scores.shape[-1])
        auroc_loss_extra_negatives = auroc(pos_premise_scores_extra_negatives, extra_neg_premise_scores)

        final_loss = tac_weight * tac_loss \
                     + pairwise_weight * (
                             pairwise_loss_positives + pairwise_loss_extra_negatives + pairwise_loss_main_negatives) \
                     + auroc_weight * ((same_goal_weight * auroc_loss_main_negatives) + auroc_loss_extra_negatives)

        return final_loss

    def forward(self, goal, premise):
        return

    def training_step(self, batch, batch_idx):
        goals, true_tacs, pos_thms, neg_thms, extra_neg_thms = batch

        try:
            tac_preds, pos_scores, neg_scores, extra_neg_scores = self(goals, pos_thms, neg_thms, extra_neg_thms)
        except Exception as e:
            print(traceback.print_exc())
            print(f"Error in forward: {e}")
            return

        loss = self.loss_func(tac_preds, true_tacs, pos_scores, neg_scores, extra_neg_scores)
        self.log("loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def backward(self, loss, *args, **kwargs) -> None:
        try:
            loss.backward()
        except Exception as e:
            print(f"Error in backward: {e}")

from code.GNNs.gnn_trainer import GNNTrainer
from code.utils import set_seed
from code.config import cfg, update_cfg
from code.data_utils.utils import check_gnn_predictions


def main(cfg):
    set_seed(cfg.seed)

    gnn_trainer = GNNTrainer(cfg=cfg)
    gnn_trainer.train_and_eval()


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    if check_gnn_predictions(
        dataset_name=cfg.dataset, gnn_model_name=cfg.gnn.model.name,
        feature=cfg.data.feature, lm_model_name=cfg.lm.model.name, seed=cfg.seed
    ):
        print("GNN predictions already exist.")
    else:
        main(cfg)

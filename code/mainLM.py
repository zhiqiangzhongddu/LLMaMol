from code.utils import set_seed
from code.config import cfg, update_cfg
from code.LMs.lm_trainer import LMTrainer
from code.data_utils.utils import check_lm_predictions


def main(cfg):
    set_seed(cfg.seed)

    trainer = LMTrainer(cfg=cfg)
    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    cfg = update_cfg(cfg)
    if check_lm_predictions(
            dataset_name=cfg.dataset, template=cfg.data.text,
            lm_model_name=cfg.lm.model.name, seed=cfg.seed
    ):
        print("LM predictions already exist.")
    else:
        main(cfg)

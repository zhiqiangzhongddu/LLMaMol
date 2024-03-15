import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'ogbg-molbbbp'
    # Model name
    # cfg.model = 'gnn'
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether to use Demo Test mode
    cfg.demo_test = False
    # Number of samples for demo test
    cfg.num_sample = 10
    # Fix the running seed to remove randomness
    cfg.seed = 42
    # Number of runs with random init
    cfg.runs = 1
    # OPENAI_API_KEY
    cfg.OPENAI_API_KEY = "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"
    cfg.data = CN()
    cfg.gnn = CN()
    cfg.lm = CN()
    cfg.llm = CN()

    # ------------------------------------------------------------------------ #
    # Data options
    # ------------------------------------------------------------------------ #
    # Load which text as input text
    # Available options: 'raw', 'IF', 'FSC-1'
    cfg.data.text = 'raw'
    # Load which feature as input text
    # Available options: 'raw', 'IF', 'FSC-1'
    cfg.data.feature = 'raw'

    # ------------------------------------------------------------------------ #
    # GNN Model options
    # ------------------------------------------------------------------------ #
    cfg.gnn.model = CN()
    # GNN model name
    cfg.gnn.model.name = 'gin-v'
    # Number of gnn layers
    cfg.gnn.model.num_layers = 4
    # Hidden size of the model
    cfg.gnn.model.hidden_dim = 128

    # ------------------------------------------------------------------------ #
    # GNN Training options
    # ------------------------------------------------------------------------ #
    cfg.gnn.train = CN()
    # Number of samples computed once per batch per device
    cfg.gnn.train.batch_size = 32
    # Base learning rate
    cfg.gnn.train.lr = 1e-3
    # Dropout rate
    cfg.gnn.train.dropout = 0.0
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.gnn.train.weight_decay = 0.0
    # Maximal number of epochs
    cfg.gnn.train.epochs = 200
    # Number of epochs with no improvement after which training will be stopped
    cfg.gnn.train.early_stop = 50
    # L2 regularization, weight decay
    cfg.gnn.train.wd = 0.0

    # ------------------------------------------------------------------------ #
    # LM Model options
    # ------------------------------------------------------------------------ #
    cfg.lm.model = CN()
    # LM model name (microsoft/deberta-base, microsoft/deberta-v3-base)
    cfg.lm.model.name = 'microsoft/deberta-base'
    cfg.lm.model.feat_shrink = ""

    # ------------------------------------------------------------------------ #
    # LM Training options
    # ------------------------------------------------------------------------ #
    cfg.lm.train = CN()
    # Number of samples computed once per batch per device
    cfg.lm.train.batch_size = 9
    # Number of training steps for which the gradients should be accumulated
    cfg.lm.train.grad_acc_steps = 1
    # Base learning rate
    cfg.lm.train.lr = 2e-5
    # Maximal number of epochs
    cfg.lm.train.epochs = 4
    # The number of warmup steps
    cfg.lm.train.warmup_epochs = 0.6
    # Number of update steps between two evaluations
    cfg.lm.train.eval_patience = 50000
    # The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights
    cfg.lm.train.weight_decay = 0.0
    # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler
    cfg.lm.train.dropout = 0.3
    # The dropout ratio for the attention probabilities
    cfg.lm.train.att_dropout = 0.1
    # The dropout ratio for the classifier
    cfg.lm.train.cla_dropout = 0.4

    # ------------------------------------------------------------------------ #
    # LLM provider options
    # ------------------------------------------------------------------------ #
    cfg.llm.provider = 'openai'
    # # The role of LLM, e.g., predictor or corrector
    # cfg.llm.role = "predictor"
    # Template name
    cfg.llm.template = 'raw'
    cfg.llm.operate = "gnn"

    # ------------------------------------------------------------------------ #
    # LLM Model options
    # ------------------------------------------------------------------------ #
    cfg.llm.model = CN()
    # LLM model name
    cfg.llm.model.name = "gpt-3.5-turbo-1106"
    cfg.llm.model.temperature = 1.
    cfg.llm.model.top_p = 1.
    cfg.llm.model.frequency_penalty = 0.
    cfg.llm.model.presence_penalty = 0.
    # temperature: Defaults to 1 (suggest 0.6?)
    #               What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
    #               make the output more random, while lower values like 0.2 will make it more
    #               focused and deterministic.
    # top_p: Defaults to 1 (suggest 0.9?)
    #               An alternative to sampling with temperature, called nucleus sampling, where the
    #               model considers the results of the tokens with top_p probability mass. So 0.1
    #               means only the tokens comprising the top 10% probability mass are considered.
    #               We generally recommend altering this or `top_p` but not both.
    # frequency_penalty: Defaults to 0
    #               Number between -2.0 and 2.0. Positive values penalize new tokens based on their
    #               existing frequency in the text so far, decreasing the model's likelihood to
    #               repeat the same line verbatim.
    # presence_penalty: Defaults to 0
    #               Number between -2.0 and 2.0. Positive values penalize new tokens based on
    #               whether they appear in the text so far, increasing the model's likelihood to
    #               talk about new topics.
    #               [See more information about frequency and presence penalties.]
    #               (https://platform.openai.com/docs/guides/text-generation/parameter-details)

    # ------------------------------------------------------------------------ #
    # LLM RaG options
    # ------------------------------------------------------------------------ #
    cfg.llm.rag = CN()
    # Template name
    cfg.llm.rag.add_knowledge = None
    # Warm up LLM
    cfg.llm.rag.warm_up = None

    # ------------------------------------------------------------------------ #
    # LLM Embed options
    # ------------------------------------------------------------------------ #
    cfg.llm.emb = CN()
    cfg.llm.emb.model = "text-embedding-3-large" # text-embedding-3-large
    cfg.llm.emb.data = "des"

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())


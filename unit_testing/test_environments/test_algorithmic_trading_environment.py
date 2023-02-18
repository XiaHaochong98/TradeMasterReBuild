from pathlib import Path
import sys
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
import argparse
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
from trademaster.environments.builder import build_environment
from trademaster.datasets.builder import build_dataset
from trademaster.environments.algorithmic_trading import AlgorithmicTradingEnvironment


def parse_args():
    parser = argparse.ArgumentParser(description='Download Alpaca Datasets')
    parser.add_argument("--config", default=osp.join(ROOT, "configs", "algorithmic_trading", "dqn_btc.py"),
                        help="download datasets config file path")
    args = parser.parse_args()
    return args

def test_algorithmic_trading_environment():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg = replace_cfg_vals(cfg)
    print(cfg)

    dataset = build_dataset(cfg)

    environment = build_environment(cfg, default_args=dict(dataset = dataset,
                                                           task = "train"))
    assert isinstance(environment, AlgorithmicTradingEnvironment)

if __name__ == '__main__':
    test_algorithmic_trading_environment()
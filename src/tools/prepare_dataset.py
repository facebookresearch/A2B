# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 


from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def merge_lists(*args):
    ret = []
    for arg in args:
        ret.extend(arg)
    return ret


OmegaConf.register_new_resolver("merge", merge_lists, replace=True)
OmegaConf.register_new_resolver(
    "range", lambda start, end, step: [*range(start, end + 1, step)], replace=True
)


base_config = OmegaConf.create(
    {
        "config_name": "default",
        "config_dir": "configs/data/",
        "out_dir": str(Path.cwd() / f"prepared_dataset"),
        "data": {"sampling_rate": 48000},
    }
)


def main(datasets_config: DictConfig):
    ds = hydra.utils.instantiate(datasets_config, _recursive_=False)
    ds.dump_to_file(datasets_config.out_dir)
    return len(ds)


if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(base_config, conf_cli)
    conf_yaml = OmegaConf.load(f"{conf.config_dir}/{conf.config_name}.yaml")
    conf = OmegaConf.merge(conf, conf_yaml, conf_cli)
    # add output dir and sampling rate
    # for validation
    out_dir = Path(conf.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conf.data.validation.out_dir = str(out_dir / f"validation")
    lval = main(datasets_config=conf.data.validation)
    print(f'Validation Dataset is exported to: "{conf.data.validation.out_dir}/"')
    print(f"Dataset chunks:      {lval}")
    # for train
    conf.data.training.out_dir = str(out_dir / f"training")
    ltrain = main(datasets_config=conf.data.training)
    print(f'Training Dataset is exported to: "{conf.data.training.out_dir}/"')
    print(f"Dataset chunks:      {ltrain}")
    # Dump experiment configurations for reproducibility
    conf.train_size, conf.validation_size = ltrain, lval
    with open(str(out_dir / f"cfg.yaml"), "w") as yaml_file:
        yaml_file.write(OmegaConf.to_yaml(conf))

# Usage: python ./tools/prepare_dataset.py config_name="urbansounds_ambix_binaural" out_dir="/mnt/home/idgebru/exported_a2b_datasets_icassp/urbansounds"

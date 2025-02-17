<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

# Ambisonic to Binaural Rendering using Neural Network
Welcome to the official repository for our ICASSP2025 paper "A2B: Ambisonic to Binaural Rendering using Neural Network."

Here you'll find the implementation code, pre-trained models, and links to the A2B dataset discussed in our paper.

# A2B Dataset
We are releasing over X hours of paired ambisonic-binaural recordings collected with a 10th order ambisonic microphone array. We've provided the microphone geometric configuration, which is required for DSP methods such as MagLS that we used as a baseline in this paper.

# Compose a dataset for model training
This allows you to combine different recordings to create a dataset that you can use for training and validation. Example configuration can be found in "configs/data/debug.yaml".

Here is an example that uses the <a href="src/config/data/debug.yaml">debug.yaml</a> configuration. It writes a ready-to-use dataset to a directory given by the out_dir cli parameter. This step writes json configuration files that will be read by a pytorch dataset loader.

``` SHELL
$ python ./tools/prepare_dataset.py config_name="n2s_mk128_binaural" out_dir="exported_speakeasy_datasets/debug/"
```

## Public datasets
We benchmarked the proposed method on publicly available ambisonic-binaural datasets. The datasets are listed below. We have added a script to download the datasets from their source.

For Urbansounds
``` SHELL
$ sh src/preprocessing/urbansounds/download.sh
```

For BTPAB
``` SHELL
$ sh src/preprocessing/bytedance/download.sh
```

# Dataset Loading
TBA

# Model Training
Please change the file paths accordingly or override from CLI
## BTPAB
```shell
$ config_name="a2b_model_bytedance_v10_1"
python ./tools/train.py config_name="models/${config_name}"
```
## Urbansounds
```shell
$ config_name="a2b_model_urbansounds_v2"
python ./tools/train.py config_name="models/${config_name}"
```

## A2B R1
```shell
$ config_name="a2b_model_n2s_mk128_v1"
python ./tools/train.py config_name="models/${config_name}"
```

## A2B R2
```shell
$ config_name="a2b_model_hearsay_mk128_v1.yaml"
python ./tools/train.py config_name="models/${config_name}"
```

# Inference and Evaluations
```shell
config_name="n2s_mk128"
python inference/evaluations.py config_name=$config_name ckpt_path="pretrained_models/a2b_n2s/checkpoints/last.ckpt"
```

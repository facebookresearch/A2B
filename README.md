<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" /></a> This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
<p align="center">
    <h1 align="center">
        A2B: Neural Rendering of Ambisonic Recordings to Binaural
    </h1>

  <p align="center">
    <img src="assets/poster.png"" alt="Overview" width="75%">
  </p>
<p align="center">
    <strong>ICASSP 2025</strong>
    <br /> 
</p>
<p align="center">
  <a href='https://idgebru.com/paper/ICASSP2025_Arxiv____A2B.pdf' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/ArXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='PDF'>
  </a>
  <a href='https://isrish.github.io/a2b/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=githubpages&logoColor=white' alt='Results Page'>
  </a>
  </p>
</p>
<br />
  

Welcome to the official repository for our ICASSP2025 paper "A2B: Ambisonic to Binaural Rendering using Neural Network."
Here you'll find the implementation code, pre-trained models, and links to the A2B dataset discussed in our paper.

# A2B Dataset
We are releasing over X hours of paired ambisonic-binaural recordings collected with a 10th order ambisonic microphone array. We've provided the microphone geometric configuration, which is required for DSP methods such as MagLS that we used as a baseline in this paper.

## Download A2B Dataset
The [A2B Dataset](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/A2B/index.html) is hosted on AWS S3.
We recommend using the AWS command line interface (see [AWS CLI installation instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)).

To download the dataset run:
```
aws s3 cp --recursive --no-sign-request s3://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/A2B/ A2B/
```
or use `sync` to avoid transferring existing files:
```
aws s3 sync --no-sign-request s3://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/A2B/ A2B/
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

# Compose a dataset for model training
This allows you to combine different recordings to create a dataset that you can use for training and validation. Example configuration can be found in "configs/data/debug.yaml".

Here is an example that uses the <a href="src/config/data/debug.yaml">debug.yaml</a> configuration. It writes a ready-to-use dataset to a directory given by the out_dir cli parameter. This step writes json configuration files that will be read by a pytorch dataset loader.

``` SHELL
$ python ./tools/prepare_dataset.py config_name="n2s_mk128_binaural" out_dir="exported_speakeasy_datasets/debug/"
```



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

# Ambisoic to Binaural Rendering using Neural Network
This is also called "M2B: Marksman to Binaural" rendering


# Ambisonics Recordings
## Internal datasets
## Public datasets

# Compose a dataset for model training
This allows you to combine different recordingss to create a dataset that you can use for training and validatation.
Example configuration can be found in "configs/data/debug.yaml".

Here is an example the uses the <a href="src/config/data/debug.yaml">debug.yaml</a> configuration. It writes a ready-to use dataset to a directory given by the out_dir cli parameter. This step write json configuration files that will be read by a pytorch dataset loader.

    :::shell
    $ python ./tools/prepare_dataset.py config_name="debug" out_dir="/mnt/home/idgebru/exported_speakeasy_datasets/debug/"
    # The training dataset I'm actively using
    $ python ./tools/prepare_dataset.py config_name="speakeasy_10hrs_all_short" out_dir="/mnt/home/idgebru/exported_speakeasy_datasets/speakeasy_10hrs_all_short/"


# Dataset Loading
TBA

# Model Training
TBA



## Submit Model Training Job Using Slurm

    :::shell
    cd /mnt/home/idgebru/git/SpeakEasy/src
    sbatch slurm/submit_job_4gpu.sh c2b_model_1ch_v9.1.15.4_stable_exp

Good results after 300k steps.

# TensorBoard
    :::shell
    # run this on dgx e.g pit103-hpcc-kubenode106
    export exp_dir="/mnt/home/idgebru/speakeasy_exp_dumps/c2b_model_1ch_v9_exp/stable/" # artifacts_dir
    tensorboard --logdir=$exp_dir --port=8989 --bind_all

### Accessing tensorboard on your local machine

    :::shell
    # assuming tenosrboard is running on pit103-hpcc-kubenode106
    ssh -L 0.0.0.0:8989:localhost:8989 pit103-hpcc-kubenode106
    # http://localhost:8989/

# Inference and Evalations
TBA

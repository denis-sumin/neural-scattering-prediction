![Teaser](teaser.jpg)
*Teaser:* We propose a neural scattering compensation for 3D color printing. Comparing to a method which uses noise-free Monte Carlo
simulation our technique achieves 300× speedup in the above case while providing the same quality.
# Neural Scattering Prediction

This repository contains code and dataset for the paper [Neural Acceleration of Scattering-Aware Color 3D Printing](https://cgg.mff.cuni.cz/publications/neural-acceleration-of-scattering-aware-color-3d-printing/). 

It contains a tensorflow v1 implementation of Radiance Predicting Neural Networks (RPNN's) that is trained on a dataset of volume-to-surface-appearance pairs.
These pairs are obtained via 512spp Monte Carlo reference renderings of a set of generated colored volumes that are halftoned to 5 discrete scattering materials.

The network is trained on scattering and absorption coefficients and outputs a single-color-channel radiance value as output.

## Dataset

[https://fabnn.mpi-inf.mpg.de/datasets/2020-02-3D+thin-generated.zip](https://fabnn.mpi-inf.mpg.de/datasets/2020-02-3D+thin-generated.zip)

To download, unzip, remove the downloaded file, do:

```bash
wget https://fabnn.mpi-inf.mpg.de/datasets/2020-02-3D+thin-generated.zip
unzip 2020-02-3D+thin-generated.zip -d data/datasets
rm 2020-02-3D+thin-generated.zip
```

## Pretrained models

| Paper model name          | Model ID | URL
| ------------------------- | :------: | -----------
| Baseline                  | m24      | https://fabnn.mpi-inf.mpg.de/models/2020_05_21_m24-3d-2ch_dataset_2020-02-3D+thin-generated.zip
| with ≡L (levels-sharing)  | m33      | https://fabnn.mpi-inf.mpg.de/models/2020_05_21_m33-3d-2ch_dataset_2020-02-3D+thin-generated.zip
| with ≡O (octants-sharing) | m30      | https://fabnn.mpi-inf.mpg.de/models/2020_05_21_m30-3d-2ch_dataset_2020-02-3D+thin-generated.zip
| with ≡L and ≡O (ours)     | m34      | https://fabnn.mpi-inf.mpg.de/models/2020_05_21_m34-3d-2ch_dataset_2020-02-3D+thin-generated.zip

## Install

This repository aims to publish the train-predict procedure described in the paper.

We suggest using Docker environment to run the training and the inference of the network 
to avoid unnecessary installation efforts.

To visualize the dataset items and the prediction results we publish a custom viewer
of OpenVDB grids which we operate on in our projects. As the viewer cannot be run in 
a headless fashion, this part of the installation should be done on the host system.

### Docker

To build the container, simply run:

```bash
docker build -t neural-scattering-prediction:latest .
```

To run a sample inference on dataset `-d` with model `-n` and saving results to the folder `-o`, run:

```bash
docker run --rm \
    --gpus all \
    --mount type=bind,source="$(pwd)/data",target=/project/data \
    neural-scattering-prediction:latest \
    python fabnn/predict_dataset.py \
        -d data/datasets/2020-02-3D+thin-generated/dataset_sample_inference.yml \
        -n data/models/2020_05_21_m34-3d-2ch_dataset_2020-02-3D+thin-generated \
        -o data/results
```

To train a model on dataset `-d` with config `-c` saving the result model to folder `-o`, run:

```bash
docker run --rm \
    --gpus all \
    --mount type=bind,source="$(pwd)/data",target=/project/data \
    neural-scattering-prediction:latest \
    python fabnn/train.py \
        -d data/datasets/2020-02-3D+thin-generated/dataset.yml \
        -c config/train_nn_prediction_config_3d_radial_symmetry_4_level_shared.json \
        -o data/models/test_model \
        -bb 50000 \
        -e 1000 \
        --epochs 500 \
        --memory-limit .6 \
        --validation-freq 10 \
        --validation-patches 50000
```

The relation of configs to the described-in-the-paper models is:

| Paper model name          | Config file 
| ------------------------- | -----------
| Baseline                  | `train_nn_prediction_config_3d_first_cubic.json`                    
| with ≡L (levels-sharing)  | `train_nn_prediction_config_3d_level_shared.json`                   
| with ≡O (octants-sharing) | `train_nn_prediction_config_3d_radial_symmetry_2.json`             
| with ≡L and ≡O (ours)     | `train_nn_prediction_config_3d_radial_symmetry_4_level_shared.json`

### VDB Viewer

To run [our viewer](vdb_view.py) of OpenVDB volumes, one should install on the host machine:

* OpenVDB with Python bindings, version >= 5.0
* [nanogui](https://github.com/wjakob/nanogui) (old version, it will **not work** with the [new nanogui](https://github.com/mitsuba-renderer/nanogui) out of the box)

#### Installing on a Ubuntu 20.04 LTS machine:

```
sudo apt install git python3-openvdb cmake xorg-dev libglu1-mesa-dev python3-dev python3-pip python3-venv
python3 -m venv --system-site-packages ~/nanogui_venv/

git clone --recursive https://github.com/wjakob/nanogui
mkdir nanogui/build && cd nanogui/build && cmake -DCMAKE_INSTALL_PREFIX="~/nanogui_venv/" ..
make install

ln -s ~/nanogui_venv/lib/nanogui.cpython-38-x86_64-linux-gnu.so ~/nanogui_venv/lib/python3.8/site-packages
export LD_LIBRARY_PATH=~/nanogui_venv/lib/

source ~/nanogui_venv/bin/activate
pip3 install pyscreenshot numpy

python3 vdb_view.py --help <my_openvdb_file.vdb>
```
**Known issues**:
 - jemalloc: [Fixed](https://jira.aswf.io/browse/OVDB-134) using `LD_PRELOAD=/path/to/libjemalloc.so python3`

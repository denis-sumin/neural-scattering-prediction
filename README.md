# neural-scattering-prediction

This repository contains code and dataset for the paper [Neural Acceleration of Scattering-Aware Color 3D Printing](https://cgg.mff.cuni.cz/publications/neural-acceleration-of-scattering-aware-color-3d-printing/). 

## Dataset



## Install

This repository aims to publish the train-predict procedure described in the paper.

We suggest using Docker environment to run the training and the inference of the network 
to avoid unnecessary installation efforts.

To visualize the dataset items and the prediction results we publish a custom viewer
of OpenVDB grids which we operate on in our projects. As the viewer cannot be run in 
a headless fashion, this part of the installation should be done on the host system.

### Docker

To build the container, simply run:

```
docker build -t neural-scattering-prediction:latest .
```

To train a model, run:


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

## Cite

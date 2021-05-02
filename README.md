# neural-scattering-prediction

This repository contains code and dataset for the paper XXX. 

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

To run our viewer of OpenVDB volumes, one should install on the host machine:

* OpenVDB with Python bindings, version >= 5.0
* nanogui


## Cite

# Score based generative model hands-on


## Introduction

This repository is inspired from the work done in the paper _Score-Based Generative Modeling through Stochastic Differential Equations_ by Song et al, and the especially the [repository](https://github.com/yang-song/score_sde_pytorch.git)

## Reference

```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

## Setup

### Getting the code

Clone the repository:

```sh
git clone https://github.com/greg2451/score-based-generative-modeling.git
```

### Configuration

1. Install conda locally following this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).
   We recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html), it is more lightweight and will be sufficient for our usage.
2. Create a new conda environment by executing the following command in your terminal:

   ```sh
   conda create -n score_generative python=3.8
   conda activate score_generative
   conda install pip
   ```

3. Having the conda environnement activated, install the [requirements](requirements.txt):

   ```sh
   pip install -r requirements.txt
   ```

## Usage (using the CLI)

### Train a model on FashionMNIST

After installing the requirements, you can train a model on FashionMNIST by running the following command:

```sh
python main.py
```

Tune the hyperparameters by passing flags to the command line. For example, to train a model with a learning rate of 0.0001, run:

```sh
python main.py --learning_rate 0.0001
```

To see all the available flags, run:

```sh
python main.py --help
```

#### Expected output

The training should happen, and the saved models will be located in the `runs` folder.

### Generate images

 After training a model, you can generate images by running the following command:

 ```sh
 python generate.py --model_path <path_to_model>
 ```

Again, all hyperparameters can be tuned by passing flags to the command line. To know more about the available flags, run:

```sh
python generate.py --help
```

#### Expected output

All generated images will be saved in the `generations` folder.


## Usage (using the jupyter notebook)

### Train a model on FashionMNIST

Just run the first cell of the [notebook](score-based-generative-model/experimentation.ipynb) and wait for the training to finish! You can tune the hyperparameters by changing the values of the variables in the first cell.
The expected output is the same as the one described in the [previous section](#train-a-model-on-fashionmnist).


### Generate images

Cells 2,3 and 4 are examples of generation setup using the three different samplers, on our pretrained model. Again, you can tune the hyperparameters by changing the values of the variables in the cells.
This time, the images won't be saved, but only displayed in the notebook.

## Development setup

This section should only be necessary if you want to contribute to the project.

Start by installing the [dev-requirements](dev-requirements.txt):

```sh
pip install -r dev-requirements.txt
```

### Enabling the pre-commit hooks

Run the following command at the root of the repository:

```sh
pre-commit install
```

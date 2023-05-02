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

## Usage

### Getting the code

Clone the repository:

```sh
git clone https://github.com/greg2451/score-based-generative-model.git
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

### Train a model on FashionMNIST

After installing the requirements, you can train a model on FashionMNIST by running the following command:

```sh
python main.py
```

Tune the parameter directly in the file.

### Generate images

 After training a model, you can generate images by running the following command:

 ```sh
 python generate.py --model_path <path_to_model>
 ```

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

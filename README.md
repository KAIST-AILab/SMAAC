# Semi-Markov Afterstate Actor-Critic (SMAAC)
This repository contains the code of [Winning the L2RPN Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic](https://openreview.net/forum?id=LmUJqB1Cz8).

## Environment setting
- python >= 3.6  
- grid2op == 0.9.4 (**Important!**)
- lightsim2grid == 0.2.3 (**Manual install required.** (https://github.com/BDonnot/lightsim2grid))  

### Create conda environment
```
conda env create -f environment.yml
conda activate smaac
```

### lightsim2grid installation
```
git clone https://github.com/BDonnot/lightsim2grid.git
cd lightsim2grid
git checkout v0.2.3
git submodule init
git submodule update
make
pip install -U pybind11
pip install -U .
```

## Data download
Since chronic data is required to train or evaluate, please [Download](https://drive.google.com/file/d/15oW1Wq7d6cu6EFS2P7A0cRhyv8u_UqWA/view?usp=sharing).  
Then, replace `data/` with it.
```
cd SMAAC
rm -rf data
tar -zxvf data.tar.gz
```

## Scripts
### Train
The detail of arguments is provided in `test.py`.
```
python test.py -n=[experiment_name] -s=[seed] -c=[environment_name (5, sand, wcci)]

# Example
python test.py -n=wcci_run -s=0 -c=wcci
```

### Evaluate
The detail of arguments is provided in `evaluate.py`.
```
python evaluate.py -n=[experiment_dirname] -c=[environment_name]

# Example
python evaluate.py -n=wcci_run_0 -c=wcci
```

## Credit
Our code is based on rte-france's Grid2Op (https://github.com/rte-france/Grid2Op)

# License Information
Copyright (c) 2020 KAIST-AILab

This source code is subject to the terms of the Mozilla Public License (MPL) v2 also available [here](https://www.mozilla.org/en-US/MPL/2.0/)

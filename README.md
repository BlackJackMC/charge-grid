# charge-grid
Electric Vehicle Battery Swapping Stations Placement Optimization

### Team 
* Phan Anh Kiệt - 25127204: LackerField 
* Đặng Nguyễn Yến Phương - 25127230: dnyp222-crypto
* Phan Quốc Bảo - 25127282: BlackJackMC

## Directory
```

```

## Setup environment
**Prerequisites**: [micromamba](https://mamba.readthedocs.io/en/latest/index.html) or similar

### Install python environment

You can install them to local folder or to the default environment folder.
```bash
# Local env folder
micromamba create -p ./env -f env.yml -y

# Default env folder
micromamba create -n <your env name> -f env.yml -y
```

After that, remember to activate your env
```bash
# Local env folder
micromamba activate ./env

# Default env folder
micromamba activate <your env name>
```

### Setup project
Setup the project using the provided `pyproject.toml` file
```bash
pip install -e .
```

And you are good to go

## Setup experiment

### Setup your custom routing model
A routing model is a model that simulate customer behaviors and also calculate fitness function.

In `./src/charge_grid/models` create a new file called `your_model.py`. Your model should inherit the `BaseModel` class and override the required methods.

```python
from .base import BaseModel

class YourModel(BaseModel):
    # Setup variable, prepare anything for your route and fitness function
    def __init__(self, 
        N: int, 
        B: int, 
        C: float, 
        P: float, 
        R: list[float], 
        L: list[list[float]], 
        Z: list[float], 
        D: list[int], 
        config: dict # Extra hyperparameter can be passed in here
        ):
        ...
    # Customer behavior
    def route(x) -> list[list[int]]:
        ...
    # Fitness function
    def fitness(x) -> float:
        ...
    # API for logging
    def get_details(x) -> dict:
        ...
```

### Setup experiment with your custom routing model
The `Experiment` class helps encapsulating all the experiment process. You just need to create an instance and pass `data`, `input_path` (for logging purpose), `output_folder` (for experiment result) and the `config` dictionary (remember to pass all hyperparameter that you use in your model here).
```python
class Experiment:
    def __init__(self, 
        data, # expect a tuple in this order: (N, B, C, P, R, L, Z, D)
        experiment_name, 
        input_path, 
        output_folder, 
        config # expect some keys, check code for more info
        ):
        ...

...
your_exp = Experiment(
    ...
    config={
        'model_builder': YourModel,
        ...
    }
)

your_exp.run()
```
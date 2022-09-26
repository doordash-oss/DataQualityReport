# DataQualityReport

DataQualityReport (DQR) endeavors to quickly, both in machine and human attention time, make users aware of potential data quality issues to be addressed in Pandas Dataframes. It is a priority that generating diagnostics should be FAST (<1 minute), such that users are encouraged to use this utility without interuption to interactive development.

# Usage
## Installation

`pip install dataqualityreport`

## Quick Start

```
from dataqualityreport import dqr_table
dqr_table(my_df) # supply your own dataframe: my_df
```

## Example Table
![Image of DQR Table](sample_dqr_table.png?raw=true "Example DQR Table")

## Learn More
View the [tutorial](tutorial.ipynb).

# Developing / Contributing

DataQualityReport uses [Poetry](https://python-poetry.org/docs/basic-usage/) for python library management.

All contributors must agree to the [Contribution License Agreement](CONTRIBUTION.txt).

```
# To create a venv with required dependencies
poetry install
```

This will use the existing `poetry.lock` file to know what dependencies (including versions) should be installed.

```
# run a command within the venv
poetry run ipython

# open a shell using the venv
poetry shell
```

A list of dependencies can be found in `/pyproject.toml`

```
# add a library using the CLI
poetry add matplotib

# To update the `poetry.lock` file based on new dependencies:
poetry update
```

## Code Style and Standards
```
# Includes mypy, pydocstring, etc.
make lint

# automatically run on every `git commit`
poetry run pre-commit install

# Tests
make unittest
```

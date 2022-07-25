# pytact

Sensor-agnostic library for interfacing with visuo-tactile sensors.

Provides a variety of algorithm implementations, scripts to train custom models, and visualizations.

## Installation

To build the library, `pybind` is required:
```bash
python3 -m pip install pybind11
```

You can install this package using pip in the root directory:

```bash
python3 -m pip install .
```

## Usage

Checkout the `scripts/` folder for a variety of examples in using this package.

The scripts also provide a few useful features, such as viewing a sensor stream, recording sensor images, and training a model for predicting the depth gradient.

To find out how to use the scripts, a help flag is provided:
```
python3 scripts/<script>.py -h
```

## Adding sensors

Create a new file in `pytact/sensors` and implement the `Sensor` base class.

## Adding tasks

Create a new file in `pytact/tasks` and implement the `Task` base class.

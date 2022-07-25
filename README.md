# pytact

Sensor-agnostic library for interfacing with visuo-tactile sensors.

Provides a variety of algorithm implementations, scripts to train custom models, and visualizations.

[Auto-documentation](https://hmccarty.github.io/pytact/)

Used in the [gelsight_ros](https://github.com/hmccarty/gelsight-ros) package.

## Installation

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

Update `get_sensor_names` and `sensor_from_args` in `pytact/sensors/util.py` to allow for dynamic creation.

## Adding tasks

Create a new file in `pytact/tasks` and implement the `Task` base class.

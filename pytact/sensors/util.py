from typing import List
from .sensors import Sensor
from .gelsight import GelsightR15

def get_sensor_names() -> List[str]:
    """
    Returns a list of string names that sensors can be referenced by.

    These will all be valid inputs to sensor_from_args().
    """
    return ["GelsightR15"]

def sensor_from_args(sensor_name: str, **kwargs) -> Sensor:
    """
    Utility function to easily create a sensor from CLI args.

    Parameters
    ----------
    sensor_name: str
        Sensor type to be created. Valid inputs are listed in get_sensor_names().
    """ 
    
    if sensor_name == "GelsightR15":
        if "url" not in kwargs or kwargs["url"] is None:
            raise KeyError("Missing required argument 'url' for GelsightR15")
        
        if "roi" in kwargs and kwargs["roi"] is not None:
            def parse_coord(coord: str):
                x, y = coord.split(',')
                return int(float(x)), int(float(y))
            roi = [parse_coord(kwargs["roi"][0]), parse_coord(kwargs["roi"][1]),
                   parse_coord(kwargs["roi"][2]), parse_coord(kwargs["roi"][3])]
            return GelsightR15(kwargs["url"], roi=roi)
        else:
            return GelsightR15(kwargs["url"])
    else:
        raise ValueError(f"Sensor name not recognized: {sensor_name}")
#!/usr/bin/env python3

import cv2
import pytact
import argparse

parser = argparse.ArgumentParser(description='Display reconstructed depth image from sensor')
parser.add_argument('sensor', type=str, choices=pytact.sensors.get_sensor_names(),
    help='Sensor type to display')
parser.add_argument('--url',  type=str, dest='url', default=None,
    help='Location of sensor stream (if needed)')
parser.add_argument('--roi', dest='roi', nargs=4, default=None,
    help='Region of interest in sensor frame, specify in order of top-left, top-right, ' +
        'bottom-right, and bottom-left. Format should be as follows: x,y x,y x,y x,y')
args = parser.parse_args()

sensor = pytact.sensors.sensor_from_args(args.sensor, **vars(args))

cv2.namedWindow('display', cv2.WINDOW_GUI_EXPANDED)
while sensor.is_running():
    frame = sensor.get_frame()
    if frame is not None:
        cv2.imshow('display', frame.image)
    if cv2.waitKey(2) == ord('q'):
        break
cv2.destroyAllWindows()

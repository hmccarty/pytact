#!/usr/bin/env python3

from ast import parse
import cv2
import pytact
import argparse

parser = argparse.ArgumentParser(description='Record a sequence of sensor images')
parser.add_argument('sensor', type=str, choices=['GelsightR15'], help='Sensor type to display')
parser.add_argument('--url',  type=str, dest='url', default='',
    help='Location of sensor stream (if needed)')
parser.add_argument('--roi', dest='roi', nargs=4,
    help='Region of interest in sensor frame, specify in order of top-left, top-right, ' +
        'bottom-right, and bottom-left. Format should be as follows: x,y x,y x,y x,y')
args = parser.parse_args()

if args.sensor == 'GelsightR15': 
    roi = None
    if args.url == '':
        print('A URL must be provided for GelsightR15')
        exit()
    elif args.roi:
        def parse_coord(coord: str):
            x, y = coord.split(',')
            return int(float(x)), int(float(y))

        roi = [parse_coord(args.roi[0]), parse_coord(args.roi[1]),
               parse_coord(args.roi[2]), parse_coord(args.roi[3])]

    sensor = pytact.sensors.GelsightR15(args.url, roi=roi)

cv2.namedWindow('display', cv2.WINDOW_GUI_EXPANDED)
while sensor.is_running():
    frame = sensor.get_frame()
    if frame is not None:
        cv2.imshow('display', frame.image)
    if cv2.waitKey(2) == ord('q'):
        break
cv2.destroyAllWindows()

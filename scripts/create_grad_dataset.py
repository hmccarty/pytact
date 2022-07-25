#!/usr/bin/env python3

import argparse
import cv2
from csv import writer
import os
import numpy as np
import pytact
from datetime import datetime as dt
import math
import random

FIELD_NAMES = ['img_name', 'R', 'G', 'B', 'x', 'y', 'gx', 'gy']

# Parse CLI arguments
parser = argparse.ArgumentParser(description="""
Utility to quickly create a pixel -> gradient dataset. This script reads from the 
folder of provided images and displays their preprocessed form accoording to the
selected sensor. On each image, an estimate of the circle deformation is labeled.
If this label is incorrect, you can redraw the label by clicking and dragging.
If you want to ignore this image, press 'n'. If you are happy with the label, you
can add the image by pressing 'y'. To finish, press 'q'.
Gradients are estimated using the provided parameters about the sphere's actual radius.
""")
parser.add_argument('input_path', type=str, help='Path to read images from')
parser.add_argument('ball_radius', type=float, help='Radius of ball used in data collection (mm)')
parser.add_argument('mmpp', type=float, help='Measure of mm per pixel')
parser.add_argument('sensor', type=str, choices=['GelsightR15'],
    help='Sensor that images were collected from')
parser.add_argument('--output_path', type=str, dest='output',
    default=os.getcwd(), help='Path to save CSV dataset to')
parser.add_argument('--amt-empty', type=float, dest='amt_empty',
    default=0.05, help='Amount of empty data points to include in dataset')
args = parser.parse_args()

# Store CLI args
radius = args.ball_radius / 1000.0
mpp = args.mmpp / 1000.0
if args.sensor == "GelsightR15":
    sensor = pytact.sensors.GelsightR15("")
else:
    print(f"Sensor type not recognized: {args.sensor}")
    exit()

# Setup dataset file
output_file = args.output + f"/data-{dt.now().strftime('%H-%M-%S')}.csv"
with open(output_file, 'w') as f:
    w = writer(f)
    w.writerow(FIELD_NAMES)

# Retrieve stored images
imgs = [args.input_path + "/" + f for f in sorted(os.listdir(args.input_path))
        if os.path.isfile(os.path.join(args.input_path, f))]

# Callback variables
current_frame = None
circle = None
click_start = None

def click_cb(event, x, y , _a, _b):
    global current_frame, circle, click_start
    if event == cv2.EVENT_LBUTTONDOWN:
        click_start = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        x_len = click_start[0] - x
        y_len = click_start[1] - y
        circle = (int(x_len/2 + x), int(y_len/2 + y), int(math.sqrt(x_len * x_len + y_len * y_len)/2))

        display_frame = current_frame.image.copy()
        cv2.circle(display_frame, (int(circle[0]), int(circle[1])),
            int(circle[2]), (0,255,0), 2)
        cv2.circle(display_frame, (int(circle[0]), int(circle[1])),
            2, (0, 0, 255), 3)

        cv2.imshow("label_data", display_frame)

# Configure cv window
cv2.namedWindow('label_data', cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback('label_data', click_cb)

while len(imgs) > 0:
    # Collect next frame and preprocess using sensor
    img = cv2.imread(imgs[0], cv2.IMREAD_COLOR)
    current_frame = pytact.types.Frame(pytact.types.FrameEnc.BGR, img)
    current_frame = sensor.preprocess_for(pytact.types.ModelType.Pixel2Grad, current_frame)

    # Convert to grayscale and find circles using hough transform
    grayscale_image = cv2.cvtColor(current_frame.image, cv2.COLOR_BGR2GRAY)
    display_image = current_frame.image.copy()

    circles = cv2.HoughCircles(grayscale_image, cv2.HOUGH_GRADIENT, 1, 20,
        param1=30,param2=30,minRadius=5,maxRadius=30)
    if circles is not None:
        for circle in circles[0]:
            cv2.circle(display_image, (int(circle[0]), int(circle[1])),
                int(circle[2]), (0,255,0), 2)
            cv2.circle(display_image, (int(circle[0]), int(circle[1])),
                2, (0, 0, 255), 3)
            break # Only print first

    cv2.imshow('label_data', display_image) 
    
    while True:
        k = cv2.waitKey(1)
        if k == ord('y'):
            if circle is None:
                print("No circle selected.")
                continue

            # Find distance in meters from circle radius
            x = np.arange(current_frame.image.shape[1])
            y = np.arange(current_frame.image.shape[0])
            xv, yv = np.meshgrid(x, y)
            gx = (circle[0] - xv) * mpp
            gy = (circle[1] - yv) * mpp
            
            # Compute x and y gradients using equation of a sphere
            dist = np.power(gx, 2) + np.power(gy, 2)
            dist_from_im = (circle[2] * mpp)**2 - dist
            dist_from_real = radius**2 - dist
            gx = np.where(dist_from_im > 0.0, -gx/np.sqrt(np.abs(dist_from_real)), 0.0)
            gy = np.where(dist_from_im > 0.0, -gy/np.sqrt(np.abs(dist_from_real)), 0.0)

            # Turn gradients into dataset labels
            labels = []
            for x in range(current_frame.image.shape[1]):
                for y in range(current_frame.image.shape[0]):
                    # Discard a certain perctage of zero gradients
                    if gx[y, x] == 0.0 and gy[y, x] == 0.0 and \
                        random.random() > args.amt_empty: 
                        continue

                    r = current_frame.image[y, x, 0]
                    g = current_frame.image[y, x, 1]
                    b = current_frame.image[y, x, 2]

                    labels.append((imgs[0], r, g, b, x, y, gx[y, x], gy[y, x]))

            # Write all labels to CSV file 
            with open(output_file, 'a', newline='') as f:
                print(f"Writing {len(labels)} labels to {output_file}")
                w = writer(f)
                for label in labels:
                    w.writerow(label)
            break
        elif k == ord('q'):
            cv2.destroyAllWindows()
            exit()
        elif k == ord('n'):
            break

    # Move to next image
    imgs = imgs[1:]

cv2.destroyAllWindows()
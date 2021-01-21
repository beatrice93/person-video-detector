# Detect humans

This script uses a pre-trained YOLOv4 network to detect people in a video; it outputs the same video with boxes drawn around each person. 

The YOLO configuration files `yolov4.weights`, `yolov4.cfg` and `coco.names` are available from AlexeyAB's [darknet](https://github.com/AlexeyAB/darknet) repository, and need to be downloaded.
If the script doesn't see the files it will download them automatically.

Requires openCV version 3 or higher.



The script takes the following arguments:
 - `-i` path to input file (required)
 - `-o` path to output file (optional)
 - `-c` path to YOLOv4 configuration folder (optional, defaults to `yolov4_cfg/`)
 - `-d` option to display the video as it is parsed

  
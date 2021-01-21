"""
Detects humans in a video file. 
Optionally writes video in output file with yellow boxes around humans. 
Arguments:
    -i input file
    -o output file
    -c config folder with YOLO weights
"""

import cv2 as cv
import numpy as np
import argparse
import os
import sys
import wget


def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i", "--input", type=str, help="Path to input video file")
    arg_parse.add_argument("-o", "--output", default=None, type=str, help="Path to output video file (optional)")
    arg_parse.add_argument("-c", "--config_folder", default=None, type=str, help="Path to YOLO configuration folder (optional)")
    arg_parse.add_argument("-d", "--display", action="store_true", help="Optional; displays the video as it is parsed")
    args = vars(arg_parse.parse_args())

    return args


def progress(count, total, status=''):
    """
    Displays a progress bar.
    """
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '.' * (bar_len - filled_len)

    sys.stdout.write('%s [%s] %s/%s  (%s %s)\r' % (status, bar, count, total, percents, '%'))
    sys.stdout.flush()


def check_config(args):
    """
    Checks whether configuration files are present in configuration folder.
    If not, downloads files.
    """
    yolo_folder = ""
    if args["config_folder"] is not None:
        yolo_folder = args["config_folder"] + "/"
    
    print("Checking configuration files in " + yolo_folder + "...")
    if not os.path.isfile(yolo_folder + "yolov4.weights"):
        wget.download("https://pjreddie.com/media/files/yolov4.weights",
                      out=yolo_folder + "yolov4.weights")
    if not os.path.isfile(yolo_folder + "coco.names"):
        wget.download("https://github.com/pjreddie/darknet/blob/master/data/coco.names",
                     out=yolo_folder + "coco.names" )
    if not os.path.isfile(yolo_folder + "yolov4.cfg"):
        wget.download("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov4.cfg",
                     out=yolo_folder + "yolov4.cfg")
    print("Done.\n")

    
def detect(frame, net, output_layers):
    """
    Detects the presence of humans in an image.
    Returns image with a yellow box around humans.
    """
    class_ids = []
    confidences = []
    boxes = []
    width = frame.shape[1]
    height = frame.shape[0]
    
    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    for out in outputs:
        for detection in out:
            # detection format: [center_x, center_y, width, height, prediction_scores]
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id == 0:
                confidence = scores[class_id]
                if confidence > 0.1:
                    #scale height and width back to the image original shape
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

    for i in indices:
        i = i[0]
        box = boxes[i]
        if class_ids[i] == 0:
            cv.rectangle(frame, (box[0],box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 255), 2)  
            
    return frame

def detect_humans(args):
    """
    Parses a video and detects humans. Optionally, writes the video in output file.
    """
    
    # initialize model with weights
    print("Loading model...")
    yolo_folder = ""
    if args["config_folder"] is not None:
        yolo_folder = args["config_folder"] + "/"
    classes = None
    with open(yolo_folder + 'coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv.dnn.readNet(yolo_folder + 'yolov4.weights', yolo_folder + 'yolov4.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("Done.\n")
    
    # parse video
    path = args["input"]
    output = args["output"]
    
    video = cv.VideoCapture(path)
    fps = video.get(cv.CAP_PROP_FPS)
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    
    print('Detecting people...')
    # do we want to write the output?
    write = False
    if args["output"] is not None:
        write = True
        out = cv.VideoWriter(args["output"], cv.VideoWriter_fourcc(*'MJPG'), fps,
                            (width, height))
        
    i = 0
    while video.isOpened():
        progress(i, total_frames, status='Frames parsed')
        check, frame = video.read() 
        if check:
            frame = detect(frame, net, output_layers) 
            if write:
                out.write(frame)
            if args["display"]:    
                cv.imshow('video', frame)    
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
        else:
            break
        i += 1
        
            
    video.release()
    if write:
        out.release
    cv.destroyAllWindows()
    print("Done.")
    


if __name__ == "__main__":
    args = argsParser()
    check_config(args)
    detect_humans(args)


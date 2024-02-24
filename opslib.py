import joblib
import os
import numpy as np
import pandas as pd
import logging
import json
import torch 
from model.model import LSTMModel
from vis.processor import Processor
import cv2
from default_params import *
from flask import jsonify
from helpers import pop_and_add, last_ip, dist, move_figure, get_hist
from vis.visual import write_on_image, visualise, activity_dict, visualise_tracking
import time
logging.basicConfig(level=logging.INFO)
cwd = os.path.abspath(os.path.dirname(__file__))

def load_model():
    """
    Load the pre-trained machine learning model
    Returns:
        model: The loaded machine learning model
    """
    model = LSTMModel(h_RNN=48, h_RNN_layers=2, drop_p=0.1, num_classes=7)
    model.load_state_dict(torch.load('model/lstm_weights.sav',map_location='cpu'))
    return model

def formatting(prediction):
    """Round the prediction to the nearest integer."""
    return np.around(prediction, 0) 
def resize(img, height, width, resolution):
    # Resize the video
    width_height = (int(width * resolution // 16) * 16,
                    int(height * resolution // 16) * 16)
    return width, height, width_height
def get_all_features(ip_set, lstm_set, model):
    valid_idxs = []
    invalid_idxs = []
    predictions = [15]*len(ip_set)  # 15 is the tag for None

    for i, ips in enumerate(ip_set):
        # ip set for a particular person
        last1 = None
        last2 = None
        for j in range(-2, -1*DEFAULT_CONSEC_FRAMES - 1, -1):
            if ips[j] is not None:
                if last1 is None:
                    last1 = j
                elif last2 is None:
                    last2 = j
        if ips[-1] is None:
            invalid_idxs.append(i)
            # continue
        else:
            ips[-1]["features"] = {}
            # get re, gf, angle, bounding box ratio, ratio derivative
            ips[-1]["features"]["height_bbox"] = get_height_bbox(ips[-1])
            ips[-1]["features"]["ratio_bbox"] = FEATURE_SCALAR["ratio_bbox"]*get_ratio_bbox(ips[-1])

            body_vector = ips[-1]["keypoints"]["N"] - ips[-1]["keypoints"]["B"]
            ips[-1]["features"]["angle_vertical"] = FEATURE_SCALAR["angle_vertical"]*get_angle_vertical(body_vector)
            # print(ips[-1]["features"]["angle_vertical"])
            ips[-1]["features"]["log_angle"] = FEATURE_SCALAR["log_angle"]*np.log(1 + np.abs(ips[-1]["features"]["angle_vertical"]))

            if last1 is None:
                invalid_idxs.append(i)
                # continue
            else:
                ips[-1]["features"]["re"] = FEATURE_SCALAR["re"]*get_rot_energy(ips[last1], ips[-1])
                ips[-1]["features"]["ratio_derivative"] = FEATURE_SCALAR["ratio_derivative"]*get_ratio_derivative(ips[last1], ips[-1])
                if last2 is None:
                    invalid_idxs.append(i)
                    # continue
                else:
                    ips[-1]["features"]["gf"] = get_gf(ips[last2], ips[last1], ips[-1])
                    valid_idxs.append(i)

        xdata = []
        if ips[-1] is None:
            if last1 is None:
                xdata = [0]*len(FEATURE_LIST)
            else:
                for feat in FEATURE_LIST[:FRAME_FEATURES]:
                    xdata.append(ips[last1]["features"][feat])
                xdata += [0]*(len(FEATURE_LIST)-FRAME_FEATURES)
        else:
            for feat in FEATURE_LIST:
                if feat in ips[-1]["features"]:
                    xdata.append(ips[-1]["features"][feat])
                else:
                    xdata.append(0)

        xdata = torch.Tensor(xdata).view(-1, 1, 5)
        # what is ips[-2] is none
        outputs, lstm_set[i][0] = model(xdata, lstm_set[i][0])
        if i == 0:
            prediction = torch.max(outputs.data, 1)[1][0].item()
            confidence = torch.max(outputs.data, 1)[0][0].item()
            fpd = True
            # fpd = False
            if fpd:
                if prediction in [1, 2, 3, 5]:
                    lstm_set[i][3] -= 1
                    lstm_set[i][3] = max(lstm_set[i][3], 0)

                    if lstm_set[i][2] < EMA_FRAMES:
                        if ips[-1] is not None:
                            lstm_set[i][2] += 1
                            lstm_set[i][1] = (lstm_set[i][1]*(lstm_set[i][2]-1) + get_height_bbox(ips[-1]))/lstm_set[i][2]
                    else:
                        if ips[-1] is not None:
                            lstm_set[i][1] = (1-EMA_BETA)*get_height_bbox(ips[-1]) + EMA_BETA*lstm_set[i][1]

                elif prediction == 0:
                    if (ips[-1] is not None and lstm_set[i][1] != 0 and \
                            abs(ips[-1]["features"]["angle_vertical"]) < math.pi/4) or confidence < 0.4:
                            # (get_height_bbox(ips[-1]) > 2*lstm_set[i][1]/3 or abs(ips[-1]["features"]["angle_vertical"]) < math.pi/4):
                        prediction = 7
                    else:
                        lstm_set[i][3] += 1
                        if lstm_set[i][3] < DEFAULT_CONSEC_FRAMES//4:
                            prediction = 7
                else:
                    lstm_set[i][3] -= 1
                    lstm_set[i][3] = max(lstm_set[i][3], 0)
            predictions[i] = prediction

    return valid_idxs, predictions[0] if len(predictions) > 0 else 15

def resize(img, height, width, resolution):
    # Resize the video
    width_height = (int(width * resolution // 16) * 16,
                    int(height * resolution // 16) * 16)
    return width, height, width_height
def get_all_features(ip_set, lstm_set, model):
    valid_idxs = []
    invalid_idxs = []
    predictions = [15]*len(ip_set)  # 15 is the tag for None

    for i, ips in enumerate(ip_set):
        # ip set for a particular person
        last1 = None
        last2 = None
        for j in range(-2, -1*DEFAULT_CONSEC_FRAMES - 1, -1):
            if ips[j] is not None:
                if last1 is None:
                    last1 = j
                elif last2 is None:
                    last2 = j
        if ips[-1] is None:
            invalid_idxs.append(i)
            # continue
        else:
            ips[-1]["features"] = {}
            # get re, gf, angle, bounding box ratio, ratio derivative
            ips[-1]["features"]["height_bbox"] = get_height_bbox(ips[-1])
            ips[-1]["features"]["ratio_bbox"] = FEATURE_SCALAR["ratio_bbox"]*get_ratio_bbox(ips[-1])

            body_vector = ips[-1]["keypoints"]["N"] - ips[-1]["keypoints"]["B"]
            ips[-1]["features"]["angle_vertical"] = FEATURE_SCALAR["angle_vertical"]*get_angle_vertical(body_vector)
            # print(ips[-1]["features"]["angle_vertical"])
            ips[-1]["features"]["log_angle"] = FEATURE_SCALAR["log_angle"]*np.log(1 + np.abs(ips[-1]["features"]["angle_vertical"]))

            if last1 is None:
                invalid_idxs.append(i)
                # continue
            else:
                ips[-1]["features"]["re"] = FEATURE_SCALAR["re"]*get_rot_energy(ips[last1], ips[-1])
                ips[-1]["features"]["ratio_derivative"] = FEATURE_SCALAR["ratio_derivative"]*get_ratio_derivative(ips[last1], ips[-1])
                if last2 is None:
                    invalid_idxs.append(i)
                    # continue
                else:
                    ips[-1]["features"]["gf"] = get_gf(ips[last2], ips[last1], ips[-1])
                    valid_idxs.append(i)

        xdata = []
        if ips[-1] is None:
            if last1 is None:
                xdata = [0]*len(FEATURE_LIST)
            else:
                for feat in FEATURE_LIST[:FRAME_FEATURES]:
                    xdata.append(ips[last1]["features"][feat])
                xdata += [0]*(len(FEATURE_LIST)-FRAME_FEATURES)
        else:
            for feat in FEATURE_LIST:
                if feat in ips[-1]["features"]:
                    xdata.append(ips[-1]["features"][feat])
                else:
                    xdata.append(0)

        xdata = torch.Tensor(xdata).view(-1, 1, 5)
        # what is ips[-2] is none
        outputs, lstm_set[i][0] = model(xdata, lstm_set[i][0])
        if i == 0:
            prediction = torch.max(outputs.data, 1)[1][0].item()
            confidence = torch.max(outputs.data, 1)[0][0].item()
            fpd = True
            # fpd = False
            if fpd:
                if prediction in [1, 2, 3, 5]:
                    lstm_set[i][3] -= 1
                    lstm_set[i][3] = max(lstm_set[i][3], 0)

                    if lstm_set[i][2] < EMA_FRAMES:
                        if ips[-1] is not None:
                            lstm_set[i][2] += 1
                            lstm_set[i][1] = (lstm_set[i][1]*(lstm_set[i][2]-1) + get_height_bbox(ips[-1]))/lstm_set[i][2]
                    else:
                        if ips[-1] is not None:
                            lstm_set[i][1] = (1-EMA_BETA)*get_height_bbox(ips[-1]) + EMA_BETA*lstm_set[i][1]

                elif prediction == 0:
                    if (ips[-1] is not None and lstm_set[i][1] != 0 and \
                            abs(ips[-1]["features"]["angle_vertical"]) < math.pi/4) or confidence < 0.4:
                            # (get_height_bbox(ips[-1]) > 2*lstm_set[i][1]/3 or abs(ips[-1]["features"]["angle_vertical"]) < math.pi/4):
                        prediction = 7
                    else:
                        lstm_set[i][3] += 1
                        if lstm_set[i][3] < DEFAULT_CONSEC_FRAMES//4:
                            prediction = 7
                else:
                    lstm_set[i][3] -= 1
                    lstm_set[i][3] = max(lstm_set[i][3], 0)
            predictions[i] = prediction

    return valid_idxs, predictions[0] if len(predictions) > 0 else 15
def predict(instance):
    """Make predictions for a list of instances"""    
    model = load_model()
    model.eval()
    ### Vertex AI reponse format ###
    
    width,height = (256,256)
    ip_sets = [[]]
    lstm_sets = [[]]
    img = cv2.resize(instance, (width, height))
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    width, height, width_height = resize(img, width,height, 1)
    processor_singleton = Processor(width_height)
    keypoint_sets, bb_list, width_height = processor_singleton.single_image(img)
    print(keypoint_sets,bb_list)
    valid1_idxs, prediction = get_all_features(ip_sets[0], lstm_sets[0], model)
    if prediction <=5:
        return True
    return False  

def generate_frames(path, frame_rate):
    video_capture = cv2.VideoCapture(path)  # Path to your video file
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Introduce a delay to control the frame rate
        time.sleep(1 / frame_rate)
def get_webcam():
    # define a video capture object 
    vid = cv2.VideoCapture(0) 
    
    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
    
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
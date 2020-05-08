import sys
import math
import time

import cv2
import numpy as np
from scipy.spatial import distance
from munkres import Munkres
from openvino.inference_engine import IENetwork, IECore

class object:
    def __init__(self, pos, feature, id=-1):
        self.feature = feature
        self.id = id
        self.time = time.monotonic()
        self.pos = pos

# DL models for pedestrian detection and person re-identification
#model_det  = 'pedestrian-detection-adas-0002'
#model_reid = 'person-reidentification-retail-0079'

# DL models for face detection and re-identification
model_det  = 'face-detection-adas-0001'
model_reid = 'face-reidentification-retail-0095'

model_det  = 'intel/' + model_det  + '/FP16/' + model_det
model_reid = 'intel/' + model_reid + '/FP16/' + model_reid

_N = 0
_C = 1
_H = 2
_W = 3

num_cameras = 2
video_caputure_list = [ 'movie1.264', 'movie2.264' ]     # for testing and debugging purpose
#video_caputure_list = [ i for i in range(num_cameras) ]  # uncomment if you want to use USB webCams

def main():
    global num_cameras
    id_num = 0
    dist_threshold = 1.0
    timeout_threshold = 5   # Object feature database timeout (sec)
    feature_db = []         # Object feature database (feature, id)

    ie = IECore()

    # Prep for face/pedestrian detection
    net_det  = ie.read_network(model_det+'.xml', model_det+'.bin')   # model=pedestrian-detection-adas-0002
    input_name_det  = next(iter(net_det.inputs))                     # Input blob name "data"
    input_shape_det = net_det.inputs[input_name_det].shape           # [1,3,384,672]
    out_name_det    = next(iter(net_det.outputs))                    # Output blob name "detection_out"
    out_shape_det   = net_det.outputs[out_name_det].shape            # [ image_id, label, conf, xmin, ymin, xmax, ymax ]
    exec_net_det    = ie.load_network(net_det, 'CPU')

    # Preparation for face/pedestrian re-identification
    net_reid = ie.read_network(model_reid+".xml", model_reid+".bin") # person-reidentificaton-retail-0079
    input_name_reid  = next(iter(net_reid.inputs))                   # Input blob name "data"
    input_shape_reid = net_reid.inputs[input_name_reid].shape        # [1,3,160,64]
    out_name_reid    = next(iter(net_reid.outputs))                  # Output blob name "embd/dim_red/conv"
    out_shape_reid   = net_reid.outputs[out_name_reid].shape         # [1,256,1,1]
    exec_net_reid    = ie.load_network(net_reid, 'CPU')

    # Open USB webcams
    caps = [cv2.VideoCapture(vcap) for vcap in video_caputure_list]

    while cv2.waitKey(1)!=27:                                 # 27 == ESC

        objects = [[] for i in range(num_cameras)]

        frames = [cap.read() for cap in caps]            # cv2.VideoCapture.read() returns [ ret, image]. Take only the image.
        images = [ frame[1] for frame in frames]
        for frame in frames:
            if frame[0]==False:
                return

        for cam in range(num_cameras):
            in_frame = cv2.resize(images[cam], (input_shape_det[_W], input_shape_det[_H]))
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape(input_shape_det)
            res_det = exec_net_det.infer(inputs={input_name_det: in_frame})     # Detect objects (pedestrian or face)

            for obj in res_det[out_name_det][0][0]:           # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
                if obj[2] > 0.6:                              # Confidence > 60% 
                    frame = images[cam]
                    xmin = abs(int(obj[3] * frame.shape[1]))
                    ymin = abs(int(obj[4] * frame.shape[0]))
                    xmax = abs(int(obj[5] * frame.shape[1]))
                    ymax = abs(int(obj[6] * frame.shape[0]))
                    class_id = int(obj[1])

                    # Obtain feature vector of the detected object using re-identification model
                    obj_img=frame[ymin:ymax,xmin:xmax]                                           # Crop the found object
                    obj_img=cv2.resize(obj_img, (input_shape_reid[_W], input_shape_reid[_H]))
                    obj_img=obj_img.transpose((2,0,1))
                    obj_img=obj_img.reshape(input_shape_reid)
                    res_reid = exec_net_reid.infer(inputs={input_name_reid: obj_img})            # Run re-identification model to generate feature vectors (256 elem)
                    
                    vec=np.array(res_reid[out_name_reid]).reshape((256))                         # Convert the feature vector to numpy array
                    objects[cam].append(object([xmin,ymin, xmax,ymax], vec))

        total_objects=0
        for obj in objects:
            total_objects += len(obj)
        if total_objects ==0:
            for i in range(num_cameras):
                cv2.imshow('cam'+str(i), images[i])
            continue

        # Create cosine distance matrix and match objects in the frame and the DB
        hangarian = Munkres()
        for cam in range(num_cameras):
            if len(feature_db)==0 or len(objects[cam])==0: continue
            dist_matrix = [ [ distance.cosine(obj_db.feature, obj_cam.feature) 
                        for obj_db in feature_db ] for obj_cam in objects[cam] ]
            combination = hangarian.compute(dist_matrix)        # Solve matching problem

            for idx_obj, idx_db in combination:
                if objects[cam][idx_obj].id!=-1:             # This object has already been assigned an ID
                    continue
                dist = distance.cosine(objects[cam][idx_obj].feature, feature_db[idx_db].feature)
                if dist < dist_threshold:
                    feature_db[idx_db].time = time.monotonic()    # Renew the last used time (extend lifetime of the DB record)
                    objects[cam][idx_obj].id = feature_db[idx_db].id
        del hangarian

        # Register remaining ID unassigned objects to the DB (They are considered as the new objects)
        for cam in range(num_cameras):
            for obj in objects[cam]:
                if obj.id == -1:
                    obj.id=id_num
                    feature_db.append(obj)
                    id_num+=1

        # Check for timeout items in the DB and delete them
        for i, db in enumerate(feature_db):
            if time.monotonic() - db.time > timeout_threshold:
                print('discarded id #{}'.format(db.id))
                feature_db.pop(i)

        # Draw bounding boxes and IDs
        for camera in range(num_cameras):
            for obj in objects[camera]:
                id = obj.id
                color = ( (((~id)<<6) & 0x100)-1, (((~id)<<7) & 0x0100)-1, (((~id)<<8) & 0x0100)-1 )
                xmin, ymin, xmax, ymax = obj.pos
                cv2.rectangle(images[camera], (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(images[camera], str(id), (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 1)
            cv2.imshow('cam'+str(camera), images[camera])

    cv2.destroyAllWindows()

if __name__ == '__main__':
        sys.exit(main() or 0)

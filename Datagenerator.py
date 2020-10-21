# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:23:49 2020

@author: kjosh
"""

import face_recognition 
from PIL import Image, ImageFile
import numpy as np
import os

def mask_face(face_landmark: dict):
    mask_img = Image.open('C:\\Users\\kjosh\\.spyder-py3\\OpenCV Projects\\Mask1.jpg').convert("RGBA")
    nose_bridge = face_landmark['nose_bridge']
    nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
    nose_v = np.array(nose_point)

    chin = face_landmark['chin']
    chin_len = len(chin)
    chin_bottom_point = chin[chin_len // 2]
    chin_bottom_v = np.array(chin_bottom_point)
    chin_left_point = chin[chin_len // 8]
    chin_right_point = chin[chin_len * 7 // 8]

    # split mask and resize
    width = mask_img.width
    height = mask_img.height
    width_ratio = 1.2
    new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

    # left
    mask_left_img = mask_img.crop((0, 0, width // 2, height))
    mask_left_width = get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
    mask_left_width = int(mask_left_width * width_ratio)
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))

    # right
    mask_right_img = mask_img.crop((width // 2, 0, width, height))
    mask_right_width = get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
    mask_right_width = int(mask_right_width * width_ratio)
    mask_right_img = mask_right_img.resize((mask_right_width, new_height))

    # merge mask
    size = (mask_left_img.width + mask_right_img.width, new_height)
    mask_img = Image.new('RGBA', size)
    mask_img.paste(mask_left_img, (0, 0), mask_left_img)
    mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

    # rotate mask
    angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
    rotated_mask_img = mask_img.rotate(angle, expand=True)

    # calculate mask location
    center_x = (nose_point[0] + chin_bottom_point[0]) // 2
    center_y = (nose_point[1] + chin_bottom_point[1]) // 2

    offset = mask_img.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

    # add mask
    face_img.paste(mask_img, (box_x, box_y), mask_img)
    
def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)
    
    

imagePaths=[]
directory = ['C:\\Users\\kjosh\\.spyder-py3\\OpenCV Projects\\Dataset\\Without mask\\']
for dir in directory:
  for filename in os.listdir(dir):
      imagePaths.append(dir+filename)
      
      
      
for image in imagePaths:
    face_image_np = face_recognition.load_image_file(image)
    face_locations = face_recognition.face_locations(face_image_np, model='hog')
    face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
    face_img = Image.fromarray(face_image_np)
    #mask_img = Image.open('C:\\Users\\kjosh\\.spyder-py3\\OpenCV Projects\\blue-mask.png')
    
    
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')
    found_face = False
    for face_landmark in face_landmarks:
        # check whether facial features meet requirement
        skip = False
        for facial_feature in KEY_FACIAL_FEATURES:
            if facial_feature not in face_landmark:
                skip = True
                break
        if skip:
            continue
    
        # mask face
        found_face = True
        mask_face(face_landmark)

    #face_img.show()
    name = os.path.basename(image)
    face_img.save('C:\\Users\\kjosh\\.spyder-py3\\OpenCV Projects\\Dataset\\masked\\'+name)
    print(name)

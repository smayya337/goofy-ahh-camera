#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Face Detection and Identification for Videos

This script can identify any humanoid face(s) within a video with some 
simple configuration. See the comments below for information on how to 
identify custom subjects. Given that computer vision tasks can be 
computationally intensive, this script optimizes performance by 
processing chunks of the video rather than the full resolution.
"""

import cv2
import numpy as np
import face_recognition

############ CONFIG SECTION ############
# enter the path of the video to be examined
video = cv2.VideoCapture('/mnt/d/Videos/Captures/2021-06-15 20-24-19.mp4')
framerate = 30 # set the frame rate of playback in frames per second (not exact)

# enter the subjects' names as keys and their portrait file paths as values
subjects = {
    'subject name here' : '/mnt/d/Pictures/Camera Roll/B7F58C12-56EA-49C7-811E-9496F44B455A_Photo.v1.png',
}

# DO NOT MODIFY ANY CODE AFTER THIS COMMENT
known_face_encodings = []
process_this_frame = False

# load and encode subject portraits as face tensors
for subject in subjects:
    portrait = face_recognition.load_image_file(subjects[subject])
    known_face_encodings.append(face_recognition.face_encodings(portrait)[0])

# press 'Q' to quit
while cv2.waitKey(1000 // framerate) & 0xFF != ord('q'):
    ret, frame = video.read()
    process_this_frame = not process_this_frame
    
    # process every other frame
    if ret and process_this_frame:
        # scale frame to 25% original size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        face_locs = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locs)
        face_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_idx = np.argmin(face_distances)[0]
            name = list(subjects.keys())[best_match_idx] if matches[best_match_idx] else "Unknown"
            face_names.append(name)
                
        # display the results
        for (top, right, bottom, left), name in zip(face_locs, face_names):
            # scale the images back to original size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # draw rectangle with label of the subject identified
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left+5, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
    cv2.imshow("Video", frame)
    
# cleanup
video.release()
cv2.destroyAllWindows()

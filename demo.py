#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:16:50 2021

# @author: jiuqiwang
# """

import numpy as np
import cv2
import mediapipe as mp
from model import zoom, normlize_range, get_CG, poses_diff, Config
from read_model import getModel

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

movements = {0: "Grab", 1: "Tap", 2: "Expand", 3: "Pinch", 4: "Rotation Clockwise", 5: "Rotation Counter-Clockwise",
             6: "Swipe Right", 7: "Swipe Left", 8: "Swipe Up", 9: "Swipe Down", 10: "Swipe X", 11: "Swipe +",
             12: "Swipe V", 13: "Shake"}


# convert marks to coordinates
def mark_to_coord(input_marks):
    coordinate = [[mark.x, mark.y, mark.z] for mark in input_marks]
    # interpolate palm
    palm = [(coordinate[0][0] + coordinate[9][0]) / 2, (coordinate[0][1] + coordinate[9][1]) / 2,
            (coordinate[0][2] + coordinate[9][2]) / 2]
    coordinate.insert(1, palm)
    return coordinate


# predict the result given the frames
def make_prediction(frames):
    p = zoom(np.copy(frames), target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
    p = normlize_range(p)
    M = get_CG(p, C)

    X_0 = np.array([M])
    X_1 = np.array([p])

    result = DD_Net.predict([X_0, X_1])[0].tolist()
    max_val = max(result)
    index = result.index(max_val)

    return index, max_val


# print the result and update the frames
def print_result(handedness, marks, frames):
    # get coordinate
    coord = mark_to_coord(marks)
    # insert into frame
    # frames = np.roll(left_frames, -66)
    frames = np.roll(frames, -66)
    frames[-1, :, :] = coord
    # predict and print the result
    index, max_val = make_prediction(frames)
    if max_val > 0.5:
        print("Handedness: " + handedness)
        print("Movement: " + movements[index])
        print("Probability: " + str(max_val))

    return frames, index


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5, max_num_hands=2) as hands:
    # set up configuration
    C = Config()
    # load DD-Net
    DD_Net = getModel()
    # create left and right frames
    left_frames = np.zeros((1024, 22, 3))
    right_frames = np.zeros((1024, 22, 3))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # describe the type of font
        # to be used.
        font = cv2.FONT_HERSHEY_SIMPLEX

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        if results.multi_hand_landmarks:
            # only one hand
            if len(results.multi_hand_landmarks) == 1:
                # get mark
                marks = results.multi_hand_landmarks[0].landmark
                # get handedness
                handedness = results.multi_handedness[0].classification[0].label

                # left hand only
                if handedness == "Left":
                    # update left frame
                    left_frames, index = print_result("Left", marks, left_frames)
                    # roll out one frame for right hand
                    right_frames = np.roll(right_frames, -66)
                    right_frames[-1, :, :] = np.zeros((22, 3))
                    cv2.putText(image, "Left:" + (movements[index]), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

                # right hand only
                else:
                    # update right frame
                    right_frames, index2 = print_result("Right", marks, right_frames)
                    # roll out one left frame
                    left_frames = np.roll(left_frames, -66)
                    left_frames[-1, :, :] = np.zeros((22, 3))
                    cv2.putText(image, "Right:" + (movements[index2]), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

                # Use putText() method for
                # inserting text on video

            # both hands
            else:
                first_marks = results.multi_hand_landmarks[0].landmark
                second_marks = results.multi_hand_landmarks[1].landmark
                first_handedness = handedness = results.multi_handedness[0].classification[0].label
                second_handedness = handedness = results.multi_handedness[1].classification[0].label

                # update both frames
                if first_handedness == "Left":
                    left_frames, index = print_result("Left", first_marks, left_frames)
                    right_frames, index2 = print_result("Right", second_marks, right_frames)

                    cv2.putText(image, ("left: " + movements[index]), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                    cv2.putText(image, ("right: " + movements[index2]), (50, 100), font, 1, (0, 255, 255), 2,
                                cv2.LINE_4)

                else:
                    right_frames, index2 = print_result("Right", first_marks, right_frames)
                    left_frames, index = print_result("Left", second_marks, left_frames)

                    # Use putText() method for
                    # inserting text on video
                    cv2.putText(image, ("left: " + movements[index]), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                    cv2.putText(image, ("right: " + movements[index2]), (50, 100), font, 1, (0, 255, 255), 2,
                                cv2.LINE_4)


        # no hand detected
        else:
            # roll out one frame for both frames
            left_frames = np.roll(left_frames, -66)
            left_frames[-1, :, :] = np.zeros((22, 3))
            right_frames = np.roll(right_frames, -66)
            right_frames[-1, :, :] = np.zeros((22, 3))

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()

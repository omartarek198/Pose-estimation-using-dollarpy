
import cv2

import mediapipe as mp

from dollarpy import Recognizer, Template, Point

import numpy as np

mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("videos/pushup2.mp4")

## Setup mediapipe instance
landmarks = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Recolor image to RGZB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)
        if results.pose_landmarks == None:
            continue


        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        lm = results.pose_landmarks.landmark
        landmarks.append( Point(lm[25].x, lm[25].y))
        landmarks.append( Point(lm[26].x, lm[26].y))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()


    tmpl_1 = Template('pushup',landmarks)



    #action2

    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    landmarks = []
    cap = cv2.VideoCapture("videos/juggling.mp4")

    ## Setup mediapipe instance
    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            landmarks.append(Point(lm[25].x, lm[25].y))
            landmarks.append(Point(lm[26].x, lm[26].y))


            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()

    tmpl_2 = Template('juggling',  landmarks)




    #action 3

    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    landmarks = []
    cap = cv2.VideoCapture("videos/pushup3.mp4")

    ## Setup mediapipe instance
    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            landmarks.append(Point(lm[25].x, lm[25].y))
            landmarks.append(Point(lm[26].x, lm[26].y))


            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()

    tmpl_3 = Template('pushup',  landmarks)






    recognizer = Recognizer([tmpl_1, tmpl_2,tmpl_3])


    #test


    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    landmarks = []
    cap = cv2.VideoCapture("videos/juggling2.mp4")

    ## Setup mediapipe instance
    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            landmarks.append(Point(lm[25].x, lm[25].y))
            landmarks.append(Point(lm[26].x, lm[26].y))


            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()


    tmpl_4 = Template('juggling',  landmarks)







    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    landmarks = []
    cap = cv2.VideoCapture("videos/juggling (2).mp4")

    ## Setup mediapipe instance
    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            print ("in")
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            landmarks.append(Point(lm[25].x, lm[25].y))
            landmarks.append(Point(lm[26].x, lm[26].y))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()

        tmpl_5 = Template('juggling', landmarks)


        #jump
        mp_drawing = mp.solutions.drawing_utils

        mp_pose = mp.solutions.pose
        landmarks = []
        cap = cv2.VideoCapture("videos/jump.mp4")

        ## Setup mediapipe instance
        landmarks = []
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                print("in")
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)
                if results.pose_landmarks == None:
                    continue
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
                lm = results.pose_landmarks.landmark
                landmarks.append(Point(lm[25].x, lm[25].y))
                landmarks.append(Point(lm[26].x, lm[26].y))

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()

            cv2.destroyAllWindows()

            tmpl_6 = Template('jumping', landmarks)


        mp_drawing = mp.solutions.drawing_utils

        mp_pose = mp.solutions.pose
        landmarks = []
        cap = cv2.VideoCapture("videos/jump2.mp4")

        ## Setup mediapipe instance
        landmarks = []
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                print("in")
                ret, frame = cap.read()
                if not ret:
                    break
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)
                if results.pose_landmarks == None:
                    continue
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
                lm = results.pose_landmarks.landmark
                landmarks.append(Point(lm[25].x, lm[25].y))
                landmarks.append(Point(lm[26].x, lm[26].y))

                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()

            cv2.destroyAllWindows()

            tmpl_7 = Template('jumping', landmarks)

        recognizer = Recognizer([tmpl_1, tmpl_2, tmpl_3, tmpl_4,tmpl_5, tmpl_6,tmpl_7])



    #test
    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    landmarks = []
    cap = cv2.VideoCapture("videos/jump3.mp4")
    print ("testing for jump")
    ## Setup mediapipe instance
    landmarks = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            landmarks.append(Point(lm[25].x, lm[25].y))
            landmarks.append(Point(lm[26].x, lm[26].y))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()


    result = recognizer.recognize(landmarks)
    print("result 1 ", result)


    #test
    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    landmarks = []
    cap = cv2.VideoCapture("videos/pushup5.mp4")

    ## Setup mediapipe instance
    landmarks = []
    print("testing for pushup")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            landmarks.append(Point(lm[25].x, lm[25].y))
            landmarks.append(Point(lm[26].x, lm[26].y))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()

    print(len(landmarks))
    result = recognizer.recognize(landmarks)
    print("result 2 " ,result)


    mp_drawing = mp.solutions.drawing_utils

    mp_pose = mp.solutions.pose
    landmarks = []
    cap = cv2.VideoCapture("videos/juggling1.mp4")

    ## Setup mediapipe instance
    landmarks = []

    print("testing for juggling")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
            if results.pose_landmarks == None:
                continue
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            lm = results.pose_landmarks.landmark
            landmarks.append(Point(lm[25].x, lm[25].y))
            landmarks.append(Point(lm[26].x, lm[26].y))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

        cv2.destroyAllWindows()


    result = recognizer.recognize(landmarks)
    print("result 2 " ,result)

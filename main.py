
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

vidcap = cv2.VideoCapture("videos/WhatsApp Video 2022-07-22 at 7.57.35 PM.mp4")
success,image = vidcap.read()
count = 0
landmarks = []
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  if (not success): break
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      results = pose.process(image)

      # Recolor back to BGR
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # Render detections
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                )
      landmarks.append(      results.pose_landmarks.landmark)
      print(landmarks[-1])


      # cv2.imshow("title", image)

      # cv2.waitKey(0)


vidcap = cv2.VideoCapture("videos/WhatsApp Video 2022-07-22 at 7.57.39 PM.mp4")
success,image = vidcap.read()
count = 0
landmarks_2 = []
while success:
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    if (not success): break
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        landmarks_2.append(results.pose_landmarks.landmark)
        print(landmarks_2[-1])





    # Import and initialize the pygame library
import pygame
pygame.init()

# Set up the drawing window
screen =pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

# Run until the user asks to quit
running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((0, 0, 0))

    # Draw a solid blue circle in the center
    pygame.draw.rect(screen, [255.,255,255], pygame.Rect(300, 200, 150, 150),2)
    pygame.draw.rect(screen, [255.,255,255], pygame.Rect(600, 200, 150, 150),2)
    for landmark in landmarks[25]:
        # pygame to draw the solid circle
        pygame.draw.circle(screen, (120, 255, 0),
                           [300 + landmark.x * 150, 200 + landmark.y * 150], 3, 3)

    for landmark in landmarks_2[25]:
        # pygame to draw the solid circle
        pygame.draw.circle(screen, (0, 255, 0),
                           [600 +landmark.x * 150, 200 +landmark.y*150 ], 3,3)


    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()




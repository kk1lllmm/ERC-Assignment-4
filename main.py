import cv2
import mediapipe as mp
import numpy as np
import time
import random as rd
import math

# Initialize webcam
#1
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH )) # To get the width of the frame
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT )) # To get the height of the frame

# Initialize hand tracking
#2
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = False , 
    max_num_hands = 1 , 
    min_detection_confidence = 0.75 , 
    min_tracking_confidence = 0.75)

# Initialize paddle and puck positions
#3
puck_coords = [int(frame_width/2) , int(frame_height/2)]
paddle_coords = [int(frame_width/2), int(frame_height)]

# Initial velocity
d = 0.8 #For damping the puck after colliding with the wall
initial_puck_velocity = [10, 10]
puck_velocity = initial_puck_velocity.copy()

# Load target image and resize it to 30,30
#4
target_img = cv2.imread(r'target.png' , cv2.IMREAD_UNCHANGED) #Change to absolute path if it throws an error
target_img = cv2.resize(target_img , (60 , 60) , interpolation = cv2.INTER_AREA) #Resizing to (60 , 60) (didn't use (30 , 30) as it was too small)

# Initialize 5 target positions randomly(remember assignment 2!!)
#5
np.random.seed(rd.randint(0 , 100))
rand_pos_x = np.random.randint(0 , frame_width - 30 , size = 5) #Generating random X axis positions for targets
rand_pos_y = np.random.randint(0 , frame_height - 30, size = 5) #Generating random Y axis positions for targets

# Initialize score
#6
is_target_hit = True
targets_hit = 0 #Number of targets hit
hit_times = [0] #Timestamps where targets are hit

score = 0

# Initialize timer variables
start_time = time.time()
game_duration = 30  # 1/2 minute in seconds

# Function to check if the puck is within a 5% acceptance region of a target
'''
def is_within_acceptance(puck, target, acceptance_percent=5):
    #complete the function
    #7
    global is_target_hit

    dist = int(math.sqrt(abs((puck[0] - target[0][targets_hit])^2 + (puck[1] - target[1][targets_hit])^2)))

    if dist >= int((frame_width/24 + 30)*(105/100)):
        is_target_hit = True
    else:
        is_target_hit = False

    return is_target_hit
'''
    
while True:

    # Calculate remaining time and elapsed time in minutes and seconds   
    #9
    current_time = time.time() #Elapsed Time
    
    # Read a frame from the webcam
    #10
    success , frame = cap.read()

    cv2.circle(frame , (int(frame_width/2) , 0) , int(frame_width/6) , (244 , 177 , 71) , 5) #Blue semi-circle on the top
    cv2.circle(frame , (int(frame_width/2) , int(frame_height)) , int(frame_width/12) , (60 , 60 , 221) , 3) #Red semi-circle on the bottom
    cv2.circle(frame , (int(frame_width/2) , int(frame_height)) , int(frame_width/48) , (60 , 60 , 221) , -1) #Smaller red semi-circle on the bottom

    # Flip the frame horizontally for a later selfie-view display
    #11
    frame.flags.writeable = False
    frame = cv2.flip(frame , 1)

    # Convert the BGR image to RGB
    #12
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    
    # Process the frame with mediapipe hands
    #13 
    result = hands.process(frame)
    frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)

    # Update paddle position based on index finger tip
    #14
    hlm = result.multi_hand_landmarks

    if hlm:
        for h in hlm:
            for id , coords in enumerate(h.landmark):

                if id == 8: #Index for tip of index finger
                    paddle_coords = [int(coords.x*frame_width ) , int(coords.y*frame_height)]
                    cv2.circle(frame , (paddle_coords[0], paddle_coords[1]) , int(frame_width/24) , (0 , 0 , 0) , -1) #Draws paddle
    
    # Update puck position based on its velocity
    #15
    if puck_velocity[0] > 0:
        puck_coords[0] += int(puck_velocity[0] * (current_time - start_time))
    else:
        puck_coords[0] += int(puck_velocity[0] * (current_time - start_time))

    if puck_velocity[1] > 0:
        puck_coords[1] += int(puck_velocity[1] * (current_time - start_time))
    else:
        puck_coords[1] += int(puck_velocity[1] * (current_time - start_time))

    cv2.circle(frame , puck_coords , int(frame_width/24) , (0 , 0 , 255) , -1) #Draws puck

    # Check for collisions with the walls
    #16
    if puck_coords[0] >= int(frame_width - frame_width/24): #Checks if puck is hitting right wall
        puck_coords[0] = int(frame_width - frame_width/24)
        puck_velocity[0] = -d * puck_velocity[0] #Damps the horizontal velocity
    if puck_coords[0] <= int(frame_width/24): #Checks if puck is hitting left wall
        puck_coords[0] = int(frame_width/24)
        puck_velocity[0] = -d * puck_velocity[0] #Damps the horizontal velocity

    if puck_coords[1] >= int(frame_height - frame_height/24): #Checks if puck is hitting ground
        puck_coords[1] = int(frame_height - frame_height/24)
        puck_velocity[1] = -d * puck_velocity[1] #Damps the vertical velocity
    if puck_coords[1] <= int(frame_height/24): #Checks if puck is hitting roof
        puck_coords[1] = int(frame_height/24)
        puck_velocity[1] = -d * puck_velocity[1] #Damps the vertical velocity

    # Check for collisions with the paddle
    #17
    div = int(math.sqrt(abs((paddle_coords[0] - puck_coords[0])**2 + (paddle_coords[1] - puck_coords[1])**2))) #Gets distance between puck and paddle
    if div <= int(frame_width/12):
        puck_velocity[1] = -puck_velocity[1] #Reverses puck's vertical velocity

    # Check for collisions with the targets(use is_within_acceptance)    
    #18
    try:
        dist1 = int(math.sqrt(abs((puck_coords[0] - rand_pos_x[targets_hit])**2 + (puck_coords[1] - rand_pos_y[targets_hit])**2))) #Gets distance between target and puck 
    except:
        dist1 = 1000 #Random value that isn't possible so that score doesn't get added needlessly
        pass

    if dist1 <= int(30*(1.25)):
        # Increase the player's score
        score += int(1000 * (1 / (current_time - hit_times[-1]))) #Adds to the score based on the time taken between subsequent target hits
        
        hit_times.append(current_time) #Adds time when target was hit to list
        
        targets_hit += 1 #Increases the value of number of targets hit

        # Increase puck velocity after each hit by 2(you will have to increase both x and y velocities)
        puck_velocity[0] += 2  #Increases X-coordinate velocity by 2 units
        puck_velocity[1] += 2  #Increases Y-coordinate velocity by 2 units

    # Draw paddle, puck, and targets on the frame and overlay target image on frame
    #19
    try:
        target_roi = frame[rand_pos_x[targets_hit] - int(60/2) : rand_pos_x[targets_hit] + int(60/2) , rand_pos_y[targets_hit] - int(60/2) : rand_pos_y[targets_hit] + int(60/2)] #Displays target image
    except:
        pass

    alpha = target_img[: , : , 3] / 255.0 #Gets alpha value of target image
    beta = 1 - alpha

    for c in range(0 , 3):
        #Lots of tries and excepts because it throws array shape errors sometimes
        try:
            target_roi[: , : , c] = (alpha * target_img[:, :, c] + beta * target_roi[:, :, c])
        except: 
            try:
                target_roi[: , : , c] = (alpha * target_img[:, :, c] + beta * target_roi[:, :, c])
            except:
                try:
                    target_roi[: , : , c] = (alpha * target_img[:, :, c] + beta * target_roi[:, :, c])
                except:
                    try:
                        target_roi[: , : , c] = (alpha * target_img[:, :, c] + beta * target_roi[:, :, c])    
                    except:
                        pass

    # Display the player's score on the frame
    #20
    cv2.putText(frame , str("Score: ") + str(score) , (30,50) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 2 , cv2.LINE_AA)
    
    # Display the remaining time on the frame
    #21
    cv2.putText(frame , str("Time: ") + str(round(30 - (current_time-start_time) , 2)) + "s", (30,90) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 0 , 0) , 2 , cv2.LINE_AA)

    # Check if all targets are hit or time is up
    #22
    if current_time - start_time >= game_duration: #Check if 30 seconds have elapsed
        cv2.rectangle(frame , (0 , frame_height) , (frame_width , 0) , (0 , 0 , 0) , -1) #Draws black background
        cv2.putText(frame , str("Score: ") + str(score) , (frame_width//2 - 75,frame_height//2) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 2 , cv2.LINE_AA) #Writes the end score of the player
        cv2.putText(frame , str("Game Over"), (frame_width//2 - 75,frame_height//2 - 75) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 2 , cv2.LINE_AA) #Writes "Game Over"

    if targets_hit >= 5: #Check if all targets are hit
        cv2.rectangle(frame , (0 , frame_height) , (frame_width , 0) , (0 , 0 , 0) , -1)
        cv2.putText(frame , str("Score: ") + str(score) , (frame_width//2 - 75,frame_height//2) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 2 , cv2.LINE_AA) #Writes the end score of the player
        cv2.putText(frame , str("You Win"), (frame_width//2 - 75,frame_height//2 - 75) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255 , 255 , 255) , 2 , cv2.LINE_AA) #Writes "You Win"

    # Display the resulting frame
    #23
    cv2.imshow("Air Hockey",frame)

    # Exit the game when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
#24
cap.release()
cv2.destroyAllWindows()

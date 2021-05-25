import cv2
import mediapipe as mp

#check 
def isCungPhia(a,b,c):
    return (a<c and b<c) or (a>c and b>c)

def abs(a):
    if a > 0:
        return a
    else:
        return -a
#ngón cái mở
def isOpened_Thumb(handlandmark):
    dx = handlandmark[0].x - handlandmark[9].x
    dy = handlandmark[0].y - handlandmark[9].y
    if abs(dy) / abs(dx) < 1:
        return isCungPhia(handlandmark[2].y, handlandmark[17].y, handlandmark[4].y)
    else:
        return isCungPhia(handlandmark[2].x, handlandmark[17].x, handlandmark[4].x)  
#ngón trỏ mở
def isOpened_Index(handlandmark):
    dx = handlandmark[0].x - handlandmark[9].x
    dy = handlandmark[0].y - handlandmark[9].y
    if abs(dy) / abs(dx) > 1:
        return isCungPhia(handlandmark[0].y, handlandmark[6].y, handlandmark[8].y)
    else:
        return isCungPhia(handlandmark[0].x, handlandmark[6].x, handlandmark[8].x)

#ngón giữa mở
def isOpened_Middle(handlandmark):
    dx = handlandmark[0].x - handlandmark[9].x
    dy = handlandmark[0].y - handlandmark[9].y
    if abs(dy) / abs(dx) > 1:
        return isCungPhia(handlandmark[0].y, handlandmark[10].y, handlandmark[12].y)
    return isCungPhia(handlandmark[0].x, handlandmark[10].x, handlandmark[12].x)
#ngón đeo nhẫn mở
def isOpened_Ring(handlandmark):
    dx = handlandmark[0].x - handlandmark[9].x
    dy = handlandmark[0].y - handlandmark[9].y
    if abs(dy) / abs(dx) > 1:
        return isCungPhia(handlandmark[0].y, handlandmark[14].y, handlandmark[16].y)
    else:
        return isCungPhia(handlandmark[0].x, handlandmark[14].x, handlandmark[16].x)
#ngón út mở
def isOpened_Pinky(handlandmark):
    dx = handlandmark[0].x - handlandmark[9].x
    dy = handlandmark[0].y - handlandmark[9].y
    if abs(dy) / abs(dx) > 1:
        return isCungPhia(handlandmark[0].y, handlandmark[18].y, handlandmark[20].y)
    else:
        return isCungPhia(handlandmark[0].x, handlandmark[18].x, handlandmark[20].x)

def countFinger(stateFinger):
    cnt = 0
    for open in stateFinger:
        if open: cnt += 1
    return cnt

def isLike(handlandmark, stateFinger):
    return handlandmark[4].y < handlandmark[1].y and stateFinger[0] and not (stateFinger[1] or 
        stateFinger[2] or stateFinger[3] or stateFinger[4])
def isFuck(handlandmark, stateFinger):
    return handlandmark[12].y < handlandmark[0].y and stateFinger[2] and not (stateFinger[0] or 
        stateFinger[1] or stateFinger[3] or stateFinger[4])

def getStateFinger(landmark):
    isOpen = []
    isOpen.append(True) if isOpened_Thumb(landmark) else isOpen.append(False)
    isOpen.append(True) if isOpened_Index(landmark) else isOpen.append(False)
    isOpen.append(True) if isOpened_Middle(landmark) else isOpen.append(False)
    isOpen.append(True) if isOpened_Ring(landmark) else isOpen.append(False)
    isOpen.append(True) if isOpened_Pinky(landmark) else isOpen.append(False)
    return isOpen

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 0.9, 0.8)

mpDraw = mp.solutions.drawing_utils

while cap.isOpened():
    success, img = cap.read()

    if not success:
        print('Ignore empty camera frame')
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img,1)

     # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    img.flags.writeable = False
    results = hands.process(img)
    img.flags.writeable = True
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            stateFinger = getStateFinger(handLms.landmark)

            cntFinger = countFinger(stateFinger)

            cv2.putText(img, 'Number: ' + str(cntFinger),
                (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)    

            #xác định cử chỉ
            gestureMessage = ''
            if isLike(handLms.landmark, stateFinger):
                gestureMessage = 'Like'
            elif isFuck(handLms.landmark, stateFinger):
                gestureMessage = 'F*ck'

            cv2.putText(img, 'Gesture: ' + gestureMessage, (10,140),
            cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
            
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # press esc to quit
    if key & 0xFF == 27:
        break
cap.release()

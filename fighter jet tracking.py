
import cv2
import numpy as np

#importing the video
cap = cv2.VideoCapture ("F22.mp4")

#setting up the video writer


hight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
result = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'),fps, (width, hight))



while True:

    ret, frame = cap.read()

    #noise reduction and enchantment
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    video = cv2.GaussianBlur(gray,(3,3),0)
    _, bw = cv2.threshold (video,110,255,cv2.THRESH_BINARY)

    #edge detection
    kernelx = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])
    kernely = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
    prewittx = cv2.filter2D(bw, -1, kernelx)
    prewitty = cv2.filter2D(bw, -1, kernely)
    prewitt = prewittx + prewitty

    #drawing the green contour
    contours, _ = cv2.findContours(prewitt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours (frame, contours, -1, (0, 255, 0), 3)

    #showing and saving the result
    cv2.imshow("bw", prewitt)
    cv2.imshow("F22", frame)
    result.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

result.release()
cap.release()
cv2.destroyAllWindows


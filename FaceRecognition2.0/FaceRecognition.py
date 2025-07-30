import cv2
from numpy import array

def face_recognition():
    model_path = "Models/deploy.prototxt"
    weight_path = "Models/res10_300x300_ssd_iter_140000.caffemodel"

    try:
        net = cv2.dnn.readNetFromCaffe(model_path, weight_path)
    except Exception as e:
        print("ERROR: ", e)
        return 1
    
    #start the camera capture
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("ERROR: Unable to access the cam")
        return 1
    
    print("press q to exit")

    #start the while loop
    while True:
        #gets one frame from the camera
        ret, frame = cam.read()

        if not ret:
            print("ERROR: Unable to read the frame")
            break


        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        faces = net.forward()

        for i in range(0, faces.shape[2]):
            conf = faces[0, 0, i, 2]
        
            if conf > 0.5:
                box = faces[0, 0, i, 3:7] * array([w, h, w, h])
                (ax, ay, bx, by) = box.astype("int")

                text = f"Face: {conf*100:.2f}%"
                y = ay - 10 if ay -10 > 10 else ay + 10
                cv2.rectangle(frame, (ax, ay), (bx, by), (255, 0, 0), 2)
                cv2.putText(frame, text, (ax, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

        #it shows the final frame and the maskedgray frame
        cv2.imshow("Face Detection", frame)
        #cv2.imshow("Gray Face Detection", grayframe)
        #if q is pressed it breaks the while loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_recognition()

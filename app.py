import cv2
from deepface import DeepFace
import time


print("======================================")
print("       AI EMOTION DETECTOR")
print("======================================")
print("Starting camera...")


# Open Mac camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    print("Check your Mac camera permissions.")
    exit()


print("Camera started successfully!")
print("Look at the camera.")
print("Press Q to quit.")


# Store the last detected emotion
current_emotion = "Starting..."

current_confidence = 0.0

last_analysis_time = 0

analysis_interval = 1.5


while True:

    # Read camera frame
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break


    # Mirror the camera
    frame = cv2.flip(frame, 1)


    # Current time
    current_time = time.time()


    # Only analyze once every 1.5 seconds
    if current_time - last_analysis_time >= analysis_interval:

        last_analysis_time = current_time

        try:

            print("Analyzing face...")


            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )


            # DeepFace can return a list
            if isinstance(result, list):
                result = result[0]


            # Get dominant emotion
            current_emotion = result["dominant_emotion"]


            # Get all emotion probabilities
            emotions = result["emotion"]


            # Get confidence
            current_confidence = emotions[current_emotion]


            print(
                f"Emotion: {current_emotion} | "
                f"Confidence: {current_confidence:.2f}%"
            )


        except Exception as error:

            # IMPORTANT:
            # Print the REAL error instead of hiding it

            print("======================================")
            print("DEEPFACE ERROR:")
            print(error)
            print("======================================")


            current_emotion = "Error"
            current_confidence = 0.0


    # ---------------------------------------
    # Display information on camera
    # ---------------------------------------

    emotion_text = current_emotion.upper()

    cv2.putText(
        frame,
        f"Emotion: {emotion_text}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {current_confidence:.2f}%",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to quit",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Show camera
    cv2.imshow(
        "AI Emotion Detector",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------
# Close everything
# ---------------------------------------

camera.release()

cv2.destroyAllWindows()

print("Camera closed.")
print("Emotion Detector stopped.")
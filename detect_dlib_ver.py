import cv2
import dlib
import numpy as np
from model import Net
import torch
from imutils import face_utils
from torchvision import transforms
import math

IMG_SIZE = (34, 26)
PATH = 'C:\\Users\\skhjh\\Desktop\\Driver-Drowsiness-Detection-master\\sleep_detect\\eye_classification\\drwosiness.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

n_count = 0

def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect

def predict(pred):
    pred = pred.transpose(1, 3).transpose(2, 3)
    outputs = model(pred)
    pred_tag = torch.round(torch.sigmoid(outputs))
    return pred_tag

def calculate_gaze_angle(left_eye, right_eye, face_center):
    # 눈의 중심 계산
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    # 얼굴 중심과 눈의 중심 간의 벡터 계산
    eye_vector = np.array([eye_center[0] - face_center[0], eye_center[1] - face_center[1]])

    # 수평 벡터
    horizontal_vector = np.array([1, 0])  # 수평 벡터 (x축 기준)

    # 벡터의 내적을 이용하여 각도 계산
    dot_product = np.dot(eye_vector, horizontal_vector)
    eye_magnitude = np.linalg.norm(eye_vector)
    horizontal_magnitude = np.linalg.norm(horizontal_vector)

    if eye_magnitude == 0 or horizontal_magnitude == 0:
        return 0

    angle = np.arccos(dot_product / (eye_magnitude * horizontal_magnitude))  # 라디안으로 각도 계산
    angle = np.degrees(angle)  # 라디안을 도로 변환
    return angle

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img_ori = cap.read()

    if not ret:
        break

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=2.5, fy=2.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        if eye_img_l is not None and eye_img_l.size > 0:
            eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        else:
            print("Left eye image is empty or invalid!")

        if eye_img_r is not None and eye_img_r.size > 0:
            eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)
        else:
            print("Right eye image is empty or invalid!")

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

        eye_input_l = torch.from_numpy(eye_input_l)
        eye_input_r = torch.from_numpy(eye_input_r)

        pred_l = predict(eye_input_l)
        pred_r = predict(eye_input_r)

        if pred_l.item() == 0.0 and pred_r.item() == 0.0:
            n_count += 1
        else:
            n_count = 0

        if n_count > 100:
            cv2.putText(img, "Wake up", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
        cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

        cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 얼굴의 중심 좌표 계산 (30번 랜드마크가 코끝)
        face_center = (shapes[30][0], shapes[30][1])

        # 시선 각도 계산
        gaze_angle = calculate_gaze_angle(shapes[36], shapes[45], face_center)

        # 시선 각도 표시
        cv2.putText(img, f"Gaze Angle: {gaze_angle:.2f}", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 화면에 표시
    cv2.imshow('result', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

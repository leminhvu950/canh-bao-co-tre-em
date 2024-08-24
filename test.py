import cv2
import numpy as np

# Đường dẫn đến các tệp mô hình
prototxt = "D:\deploy.prototxt"
model = "D:\mobilenet_iter_73000.caffemodel"

# Tải mô hình đã huấn luyện
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Danh sách các lớp có thể phát hiện
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Resize hình ảnh để tốc độ xử lý nhanh hơn
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Đưa hình ảnh vào mạng để dự đoán
    net.setInput(blob)
    detections = net.forward()

    # Lặp qua các dự đoán
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Lọc ra các dự đoán có độ tin cậy cao
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label == "person":
                # Lấy tọa độ bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Vẽ bounding box và nhãn lên frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("Frame", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()

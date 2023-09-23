"""
Kode ini memiliki tujuan untuk mendeteksi wajah dalam gambar, menganalisis usia, jenis kelamin, ekspresi, 
dan menambahkan anotasi teks pada gambar yang menunjukkan hasil analisis. 
Hasil gambar dengan anotasi akan disimpan dalam folder "results".

Kode ini menggunakan model-model yang telah dipersiapkan dalam folder "models" untuk deteksi wajah, 
usia, jenis kelamin, serta model ekspresi wajah yang telah dilatih sebelumnya. 
Proses berlangsung dengan bantuan modul "tqdm" yang menampilkan progress bar yang menarik 
selama proses berlangsung.

----------------------------------------------------------------------------------------------------------
Instruksi penggunaan:
1. Pastikan model-model terletak di dalam folder "models".
2. Panggil fungsi `detect_and_annotate_faces('nama_foto')` untuk menganalisis gambar yang diinginkan.
3. Hasil gambar dengan anotasi akan disimpan dalam folder "results".
----------------------------------------------------------------------------------------------------------
Catatan: Kode ini digunakan untuk tujuan demonstrasi dan dapat disesuaikan sesuai kebutuhan.

Oleh :  _drat (c)2023
Versi : 1.0
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mengatur level log TensorFlow ke '2' untuk menghilangkan pesan info

import cv2
import math
import numpy as np
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image

# untuk pogress bar
from tqdm import tqdm
import time

# ---------
# PATHS
# ---------
models_folder = "models"
results_folder = "results"

# Mendefinisikan jalur ke model ekspresi wajah
model_json_path = os.path.join(models_folder, "model_fer.json")
model_weights_path = os.path.join(models_folder, "model_fer.h5")

# Definisi model dan prototipe
faceProto = os.path.join(models_folder, "opencv_face_detector.pbtxt")
faceModel = os.path.join(models_folder, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(models_folder, "age_deploy.prototxt")
ageModel = os.path.join(models_folder, "age_net.caffemodel")
genderProto = os.path.join(models_folder, "gender_deploy.prototxt")
genderModel = os.path.join(models_folder, "gender_net.caffemodel")
haarcascade_smile = os.path.join(models_folder, "haarcascade_smile.xml")

# Parameter dan data series
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Pria','Wanita']

# DNN CV2
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Membaca model ekspresi wajah
model = model_from_json(open(model_json_path, "r").read())
model.load_weights(model_weights_path)
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

# Mengatur resolusi yang lebih kecil
padding = 20

#
# Fungsi untuk menyoroti wajah dalam frame
#
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Membuat blob dari frame
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    # Loop untuk mendeteksi wajah
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    
    return frameOpencvDnn, faceBoxes


#
# Deteksi Ekspresi Wajah
#
def detectExpression(face):
    # Konversi ROI wajah ke grayscale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Gunakan Cascade Classifier untuk mendeteksi senyuman
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)
    
    # Jika terdeteksi senyuman, kembalikan "Senyum", jika tidak, kembalikan "Tidak Senyum"
    if len(smiles) > 0:
        return "Senyum"
    else:
        return "Tidak Senyum"

#
# Fungsi untuk mendeteksi emosi wajah
#
def detectEmotion(face):
    # Resize gambar menjadi 48x48 piksel (sesuai dengan model FER-2013)    
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = np.reshape(face, [1, 48, 48, 1])

    # Prediksi ekspresi wajah
    emotion = model.predict(face)
    emotion_label = emotion_labels[np.argmax(emotion)]

    return emotion_label


#
# Anotasi text untuk foto yang disimpan
#
def detect_and_annotate_faces(input_image_path):
    # Mendapatkan nama file tanpa ekstensi
    file_name = os.path.splitext(os.path.basename(input_image_path))[0]

    frame = cv2.imread(input_image_path)
    resultImg, faceBoxes = highlightFace(faceNet, frame)

    # Mendapatkan ROI (Region of Interest) pada wajah
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding, frame.shape[0]-1),
                     max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
        
        # Membuat blob dari ROI
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Mendeteksi ekspresi wajah
        expression = detectExpression(face)
        # Deteksi emosi 
        emotion_det = detectEmotion(face)
        # Mendeteksi jenis kelamin
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # Mendeteksi usia
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Warna Text
        text_color = (255, 255, 255)
        background_color = (11, 112, 232)
        face_box_color = (38, 230, 232)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Menentukan posisi awal tulisan
        text_x = faceBox[2] + 10
        text_y = faceBox[1] - 30

        # Menggambar kotak di sekitar wajah dengan latar belakang biru
        cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), face_box_color, int(round(frame.shape[0]/150)), 8)
        
        # Menambahkan teks usia di sisi kanan kotak dengan latar belakang hitam
        cv2.rectangle(resultImg, (text_x - 5, text_y - 20), (text_x + 200, text_y + 10), background_color, -1)
        cv2.putText(resultImg, f'Age: {age[1:-1]} thn', (text_x, text_y), font, 0.6, text_color, 1, cv2.LINE_AA)

        # Menambahkan teks emosi di sisi kanan kotak dengan latar belakang hitam
        text_y += 30
        cv2.rectangle(resultImg, (text_x - 5, text_y - 20), (text_x + 200, text_y + 10), background_color, -1)
        cv2.putText(resultImg, f'Emotion: {emotion_det}', (text_x, text_y), font, 0.6, text_color, 1, cv2.LINE_AA)

        # Menambahkan teks jenis kelamin di sisi kanan kotak dengan latar belakang hitam
        text_y += 30
        cv2.rectangle(resultImg, (text_x - 5, text_y - 20), (text_x + 200, text_y + 10), background_color, -1)
        cv2.putText(resultImg, f'Sex: {gender}', (text_x, text_y), font, 0.6, text_color, 1, cv2.LINE_AA)

        # Menambahkan teks ekspresi di sisi kanan kotak dengan latar belakang hitam
        text_y += 30
        cv2.rectangle(resultImg, (text_x - 5, text_y - 20), (text_x + 200, text_y + 10), background_color, -1)
        cv2.putText(resultImg, f'Exp: {expression}', (text_x, text_y), font, 0.6, text_color, 1, cv2.LINE_AA)

    # Menyimpan gambar dengan kotak wajah dan teks hasil prediksi ke output_image_path dalam folder "results"
    output_image_path = os.path.join(results_folder, f'{file_name}_result.jpg')
    cv2.imwrite(output_image_path, resultImg)
#
# START
# 
if __name__ == '__main__':

    # Jumlah iterasi atau durasi proses
    total_images = 10

    # Membuat objek tqdm dengan total iterasi
    progress_bar = tqdm(total=total_images, desc="Processing Images", unit="Image")

    # Loop atau proses Anda
    for i in range(total_images):
        # Proses yang sedang berlangsung
        detect_and_annotate_faces('girl1.jpg')
        detect_and_annotate_faces('girl2.jpg')
        detect_and_annotate_faces('kid1.jpg')
        detect_and_annotate_faces('kid2.jpg')
        detect_and_annotate_faces('man1.jpg')
        detect_and_annotate_faces('man2.jpg')
        detect_and_annotate_faces('woman1.jpg')
        detect_and_annotate_faces('woman2.jpg')
        detect_and_annotate_faces('my_wife1.jpg')
        detect_and_annotate_faces('my_wife2.jpg')
        detect_and_annotate_faces('me_and_wife.jpg')
        
        # Update progress bar
        progress_bar.update(1)

    # Menutup progress bar saat selesai
    progress_bar.close()

    print("[-] Proses selesai!")


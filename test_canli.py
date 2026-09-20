import cv2
import joblib
import pandas as pd
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_YOLU = r"C:\yuz_algila\face_landmarker.task"
SINIFLANDIRICI_YOLU = r"C:\yuz_algila\yuz_ifade_modeli.pkl" 

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

def landmarker_olustur():
    secenekler = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_YOLU),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True
    )
    return FaceLandmarker.create_from_options(secenekler)

def mp_image_olustur(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

def blendshape_dict_al(result):
    if not result.face_blendshapes or len(result.face_blendshapes) == 0:
        return None
    blendshape_listesi = result.face_blendshapes[0]
    ozellikler = {}
    for kategori in blendshape_listesi:
        ozellikler[kategori.category_name] = kategori.score
    return ozellikler

def main():
    uyuma_baslangic_zamani = 0
    uyuyor_mu = False
    model = joblib.load(SINIFLANDIRICI_YOLU)
    landmarker = landmarker_olustur()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        mp_image = mp_image_olustur(frame)
        result = landmarker.detect(mp_image)
        ozellikler = blendshape_dict_al(result)

        tahmin_yazi = "Yuz bulunamadi"

        if ozellikler is not None:
            X_tek = pd.DataFrame([ozellikler])
            tahmin = model.predict(X_tek)[0]
            
            if tahmin == "uyuyor":
                if uyuma_baslangic_zamani == 0:
                    uyuma_baslangic_zamani = time.time()
                
                gecen_sure = time.time() - uyuma_baslangic_zamani
                
                if gecen_sure > 0.1: 
                    uyuyor_mu = True
                else:
                    uyuyor_mu = False
            else:
                uyuma_baslangic_zamani = 0
                uyuyor_mu = False

            if uyuyor_mu:
                tahmin_yazi = "UYUYOR (ALARM!)"
            else:
                if hasattr(model, "predict_proba"):
                    olasiliklar = model.predict_proba(X_tek)[0]
                    en_yuksek_index = olasiliklar.argmax()
                    en_yuksek_etiket = model.classes_[en_yuksek_index]
                    
                    if en_yuksek_etiket == "uyuyor":
                        tahmin_yazi = "dogal" 
                    else:
                        tahmin_yazi = f"{en_yuksek_etiket} ({olasiliklar[en_yuksek_index]:.2f})"
                else:
                    tahmin_yazi = "dogal" if tahmin == "uyuyor" else tahmin

        cv2.putText(
            frame,
            f"Tahmin: {tahmin_yazi}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if uyuyor_mu else (0, 255, 0),
            2
        )

        cv2.imshow("Canli Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
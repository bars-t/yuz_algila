import cv2
import csv
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_YOLU = r"C:\yuz_algila\face_landmarker.task"
CSV_YOLU = r"C:\yuz_algila\yuz_ifade_verisi.csv"
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

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    return mp_image


def blendshape_dict_al(result):

    if not result.face_blendshapes or len(result.face_blendshapes) == 0:
        return None

    blendshape_listesi = result.face_blendshapes[0]

    ozellikler = {}

    for kategori in blendshape_listesi:
        ozellikler[kategori.category_name] = kategori.score

    return ozellikler


def landmarklari_ciz(frame, result):
    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return frame

    h, w, _ = frame.shape
    yuz = result.face_landmarks[0]
    for nokta in yuz:
        x = int(nokta.x * w)
        y = int(nokta.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    return frame


def csv_satiri_yaz(etiket, ozellikler):
    dosya_var_mi = os.path.exists(CSV_YOLU)
    alanlar = ["label"] + list(ozellikler.keys())
    satir = {"label": etiket}
    satir.update(ozellikler)

    try:
        if not os.path.exists(CSV_YOLU):
            with open(CSV_YOLU, "w", newline="", encoding="utf-8") as f:
                alanlar = ["label"] + list(ozellikler.keys())
                yazici = csv.DictWriter(f, fieldnames=alanlar)
                yazici.writeheader()

        with open(CSV_YOLU, "a", newline="", encoding="utf-8") as f:
            alanlar = ["label"] + list(ozellikler.keys())
            yazici = csv.DictWriter(f, fieldnames=alanlar)

            satir = {"label": etiket}
            satir.update(ozellikler)

            yazici.writerow(satir)

    except PermissionError:
        print("⚠️ CSV dosyasi kilitli! Excel acik olabilir.")

def main():
    landmarker = landmarker_olustur()

    cap = cv2.VideoCapture(0)

    aktif_etiket = "dogal"


    son_kayit_zamani = 0


    kayit_araligi = 0.2

    print("Tuslar:")
    print("d -> dogal")
    print("m -> mutlu")
    print("q -> cikis")
    print("a -> sasirma")
    print("u -> uyuyor")
    print("s -> sinirli")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera goruntusu alinamadi.")
            break
        mp_image = mp_image_olustur(frame)
        result = landmarker.detect(mp_image)
        ozellikler = blendshape_dict_al(result)
        frame = landmarklari_ciz(frame, result)
        if ozellikler is not None:
            simdi = time.time()
            if simdi - son_kayit_zamani >= kayit_araligi:
                csv_satiri_yaz(aktif_etiket, ozellikler)
                son_kayit_zamani = simdi
        cv2.putText(
            frame,
            f"Aktif etiket: {aktif_etiket}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            "d:dogal m:mutlu q:cikis",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Veri Toplama", frame)

        tus = cv2.waitKey(1) & 0xFF

        if tus == ord("d"):
            aktif_etiket = "dogal"
        elif tus == ord("m"):
            aktif_etiket = "mutlu"
        elif tus == ord("q"):
            break
        elif tus == ord("u"):
            aktif_etiket = "uyuyor"
        elif tus == ord("s"):
            aktif_etiket = "sinirli"
        elif tus == ord("a"):
            aktif_etiket = "sasirma"

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
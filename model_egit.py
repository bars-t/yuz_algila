import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score
CSV_YOLU = r"C:\yuz_algila\yuz_ifade_verisi.csv"
MODEL_CIKTISI = r"C:\yuz_algila\yuz_ifade_modeli.pkl"


def main():
    df = pd.read_csv(CSV_YOLU)
    X = df.drop("label", axis=1)

    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print("Dogruluk:", accuracy_score(y_test, y_pred))
    print("\nSiniflandirma Raporu:")
    print(classification_report(y_test, y_pred))


    joblib.dump(model, MODEL_CIKTISI)


    print(f"\nModel kaydedildi: {MODEL_CIKTISI}")



if __name__ == "__main__":
    main()
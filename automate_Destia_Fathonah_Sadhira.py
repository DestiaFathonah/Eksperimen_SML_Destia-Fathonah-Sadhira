import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    df = df.dropna()
    df = df.drop_duplicates()

    categorical_cols = [
        'sex',
        'cp',
        'fbs',
        'restecg',
        'exang',
        'slope',
        'thal'
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )

    numerical_cols = [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak'
    ]

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Preprocessing selesai.")
    print(f"File disimpan di: {output_path}")


if __name__ == "__main__":
    INPUT_PATH = "namadataset_raw/heart.csv"
    OUTPUT_PATH = "preprocessing/namadataset_preprocessing/heart_preprocessed.csv"

    preprocess_data(INPUT_PATH, OUTPUT_PATH)
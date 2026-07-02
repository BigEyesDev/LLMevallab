from src.core.config import load_config
from src.pipeline.europarl_loader import EuroParlDataLoader
from src.pipeline.cnn_dailymail_loader import CNNDailyMailLoader


def download_europarl_dataset():
    config = load_config()
    ds = config["datasets"]["europarl"]
    loader = EuroParlDataLoader(sample_size=ds["sample_size"])
    path = loader.download_and_prepare(ds["language_pair"])
    print(f"Dataset saved to: {path}")


def download_cnn_dailymail_dataset():
    config = load_config()
    ds = config["datasets"]["cnn_dailymail"]
    loader = CNNDailyMailLoader(sample_size=ds["sample_size"])
    path = loader.download_and_prepare()
    print(f"CNN/DailyMail saved to: {path}")


def data_loading():
    print("--------------------------------")
    print("Downloading Europarl dataset")
    print("--------------------------------")
    download_europarl_dataset()
    print("--------------------------------")
    print("Downloading CNN/DailyMail dataset")
    print("--------------------------------")
    download_cnn_dailymail_dataset()


def main():
    print("--------------------------------")
    print("Hello from llmevallab!")
    print("--------------------------------")

    print("--------------------------------")
    print("Data loading")
    print("--------------------------------")
    data_loading()
    print("--------------------------------")
    print("Data loading complete")
    print("--------------------------------")


if __name__ == "__main__":
    main()

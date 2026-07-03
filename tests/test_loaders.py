from unittest.mock import patch

from src.pipeline.cnn_dailymail_loader import CNNDailyMailLoader
from src.pipeline.europarl_loader import EuroParlDataLoader


def _fake_europarl_examples(n: int):
    for i in range(n):
        yield {
            "translation": {
                "de": f"German text {i}",
                "en": f"English text {i}",
            }
        }


def _fake_cnn_examples(n: int):
    for i in range(n):
        yield {
            "article": f"Article {i}",
            "highlights": f"Summary {i}",
            "id": str(i),
        }


@patch("src.pipeline.europarl_loader.load_dataset")
def test_europarl_loader_respects_sample_size(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = _fake_europarl_examples(50)
    loader = EuroParlDataLoader(processed_dir=str(tmp_path), sample_size=5)
    output_path = loader.download_and_prepare("de-en")

    documents = loader.load_from_disk(str(output_path))
    assert len(documents) == 5
    mock_load_dataset.assert_called_once()


@patch("src.pipeline.cnn_dailymail_loader.load_dataset")
def test_cnn_dailymail_loader_respects_sample_size(mock_load_dataset, tmp_path):
    mock_load_dataset.return_value = _fake_cnn_examples(50)
    loader = CNNDailyMailLoader(processed_dir=str(tmp_path), sample_size=7)
    output_path = loader.download_and_prepare()

    documents = loader.load_from_disk(str(output_path))
    assert len(documents) == 7
    mock_load_dataset.assert_called_once()

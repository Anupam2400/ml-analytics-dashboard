from src.ingestion.load_data import load_data


def test_load_data():
    df = load_data()
    assert not df.empty, "DataFrame should not be empty"
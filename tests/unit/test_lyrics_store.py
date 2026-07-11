"""Tests for the song_id -> lyrics store."""


def test_get_by_row_position():
    from api.services.songs import LyricsStore

    s = LyricsStore(["first song words", "second song words"])
    assert s.get(0) == "first song words"
    assert s.get(1) == "second song words"
    assert len(s) == 2


def test_out_of_range_returns_none():
    from api.services.songs import LyricsStore

    s = LyricsStore(["only one"])
    assert s.get(5) is None
    assert s.get(-1) is None


def test_non_string_returns_none():
    from api.services.songs import LyricsStore

    s = LyricsStore([float("nan")])
    assert s.get(0) is None


def test_from_csv(tmp_path):
    import pandas as pd

    from api.services.songs import LyricsStore

    p = tmp_path / "songs.csv"
    pd.DataFrame({"lyrics": ["a b c", "d e f"], "mood": ["Sad", "Hype"]}).to_csv(p, index=False)
    s = LyricsStore.from_csv(p)
    assert len(s) == 2
    assert s.get(1) == "d e f"

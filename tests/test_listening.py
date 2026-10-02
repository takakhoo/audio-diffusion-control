import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("listening_results", Path(__file__).parents[1] / "experiments" / "listening_results.py")
listening = importlib.util.module_from_spec(spec)
spec.loader.exec_module(listening)


def session(choices, check=5):
    answers = [dict(task="direction", slider="mood", method="lora", choice=c, expected="A") for c in choices]
    answers.append(dict(task="same", check=False, slider="mood", method="lora", xB=2, same=4, quality=3))
    answers.append(dict(task="same", check=True, slider="mood", method="lora", xB=0, same=check, quality=5))
    return dict(answers=answers)


def test_summary_counts_hits_and_drops_careless_sessions():
    report = listening.summarise([session("AAAB"), session("AAAA"), session("BBBB", check=2)])
    assert report["sessions"] == 3 and report["kept"] == 2
    row = report["direction"][0]
    assert row["n"] == 8 and row["accuracy"] == 7 / 8
    assert row["low"] < 7 / 8 < row["high"] and 0.4 < row["low"] < 0.6
    assert row["p"] < 0.05
    assert report["edits"][0]["same"] == 4 and report["edits"][0]["quality"] == 3


def test_interval_handles_all_or_nothing():
    assert listening.interval(0, 10)[0] == 0.0 and listening.interval(10, 10)[1] == 1.0
    assert listening.interval(10, 10)[0] > 0.65

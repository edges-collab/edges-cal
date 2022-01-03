from pathlib import Path

from edges_cal import CalibrationObservation
from edges_cal.simulate import simulate_q_from_calobs


def test_simulate_q(cal_data: Path):
    obs = CalibrationObservation(cal_data)

    q = simulate_q_from_calobs(obs, "open")
    assert len(q) == obs.freq.n

import numpy as np
from pathlib import Path

from edges_cal import CalibrationObservation
from edges_cal.simulate import simulate_q_from_calobs


def test_simulate_q(cal_data: Path):
    obs = CalibrationObservation(cal_data)

    q = simulate_q_from_calobs(
        obs, ant_s11=1, ant_temp=np.linspace(1, 10, obs.freq.n) ** -2.5
    )
    assert len(q) == obs.freq.n

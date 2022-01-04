import numpy as np
from pathlib import Path

from edges_cal import CalibrationObservation
from edges_cal.simulate import simulate_q_from_calobs, simulate_qant_from_calobs


def test_simulate_q(cal_data: Path):
    obs = CalibrationObservation(cal_data)

    q = simulate_q_from_calobs(obs, "open")
    qhot = simulate_q_from_calobs(obs, "hot_load", freq=obs.freq.freq)

    assert len(q) == obs.freq.n == len(qhot)
    assert not np.all(q == qhot)

    obsc = obs.to_calfile()

    q2 = simulate_q_from_calobs(obsc, "open")
    np.testing.assert_allclose(q, q2)


def test_simulate_qant(cal_data: Path):
    obs = CalibrationObservation(cal_data)
    q = simulate_qant_from_calobs(
        obs,
        ant_s11=np.zeros(obs.freq.n),
        ant_temp=np.linspace(1, 100, obs.freq.n) ** -2.5,
    )
    assert len(q) == obs.freq.n

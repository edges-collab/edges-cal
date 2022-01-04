import numpy as np
from pathlib import Path

from edges_cal import CalibrationObservation
from edges_cal.simulate import simulate_q_from_calobs


def test_simulate_q(cal_data: Path):
    obs = CalibrationObservation(cal_data)

    q = simulate_q_from_calobs(obs, "open")
    qhot = simulate_q_from_calobs(obs, "hot_load")

    assert len(q) == obs.freq.n == len(qhot)
    assert not np.all(q == qhot)

    obsc = obs.to_calfile()

    q2 = simulate_q_from_calobs(obsc, "open")
    np.testing.assert_allclose(q, q2)

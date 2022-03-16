import numpy as np

from edges_cal import CalibrationObservation
from edges_cal.simulate import simulate_q_from_calobs, simulate_qant_from_calobs


def test_simulate_q(calobs: CalibrationObservation):
    q = simulate_q_from_calobs(calobs, "open")
    qhot = simulate_q_from_calobs(calobs, "hot_load", freq=calobs.freq.freq)

    assert len(q) == calobs.freq.n == len(qhot)
    assert not np.all(q == qhot)


def test_simulate_qant(calobs: CalibrationObservation):

    q = simulate_qant_from_calobs(
        calobs,
        ant_s11=np.zeros(calobs.freq.n),
        ant_temp=np.linspace(1, 100, calobs.freq.n) ** -2.5,
    )
    assert len(q) == calobs.freq.n

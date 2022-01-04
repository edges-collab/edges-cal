import pytest

from edges_cal import cached_property as cp


class Derp:
    @cp.safe_property
    def lard(self):
        self.non_existent
        print("did it")
        return 1

    @cp.cached_property
    def cheese(self):
        self.macaroni
        print("did it")
        return 2

    def __getattr__(self, item):
        if item == "nugget":
            return "spam"

        raise AttributeError()


def test_safe_property():
    d = Derp()
    with pytest.raises(RuntimeError, match="Wrapped AttributeError"):
        d.lard


def test_cached_property():
    d = Derp()
    with pytest.raises(RuntimeError, match="failed with an AttributeError"):
        d.cheese

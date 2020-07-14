"""A wrapper of cached_property that also saves which things are cached."""
from cached_property import cached_property as cp


class cached_property(cp):  # noqa
    def __get__(self, obj, cls):
        """Get the store value of the attribute."""
        value = super().__get__(obj, cls)

        # Add the name of the decorated func to the _cached_ item of the dict
        if obj is not None:
            if "_cached_" not in obj.__dict__:
                obj.__dict__["_cached_"] = set()

            obj.__dict__["_cached_"].add(self.func.__name__)

        return value

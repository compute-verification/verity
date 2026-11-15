import signal

import pytest


@pytest.fixture
def stop_held():
    """A SIGTERM a test sends itself that the code under test takes no handler for fails the test, rather than end
    pytest."""
    previous = signal.signal(signal.SIGTERM, lambda *_: pytest.fail("a SIGTERM no handler of the run's took"))
    yield
    signal.signal(signal.SIGTERM, previous)

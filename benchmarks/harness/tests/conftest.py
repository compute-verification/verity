"""The tests' shared stand-ins are modules beside them, whose asserts pytest rewrites as it rewrites a test's."""

import pytest

pytest.register_assert_rewrite("steady_standins")

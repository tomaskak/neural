from neural.tools.example import double

import pytest


class TestExample:
    def test_run_this_returns_two(self):
        assert double(6) == 12

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_model import encode_state, load_q_table, test_specific_state


class FakeEnv:
    def set_predefined_state(self, state):
        self.state = state
        self.actions = []
    def find_available_moves(self):
        return [1, 81]
    def step(self, action):
        self.actions.append(action)
        return self.state, 1, action == 81, {}
    def render(self):
        pass


class EvaluationTests(unittest.TestCase):
    def test_roundtrip_trusted_table(self):
        table = {(1, 0): np.arange(82)}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'table.npy'
            np.save(path, table)
            actual = load_q_table(path)
        np.testing.assert_array_equal(actual[(1, 0)], table[(1, 0)])

    def test_unknown_state_stops(self):
        env = FakeEnv()
        with contextlib.redirect_stdout(io.StringIO()):
            test_specific_state(env, {}, {'bobbin_positions': [1, 0]})
        self.assertEqual(env.actions, [81])

    def test_legal_action_and_step_limit(self):
        env = FakeEnv()
        state = {'bobbin_positions': [1, 0]}
        values = np.zeros(82)
        values[0] = 100  # invalid action must be ignored
        values[1] = 10
        with contextlib.redirect_stdout(io.StringIO()):
            reward = test_specific_state(env, {encode_state(state): values}, state, max_steps=3)
        self.assertEqual(env.actions, [1, 1, 1])
        self.assertEqual(reward, 3)


if __name__ == '__main__':
    unittest.main()

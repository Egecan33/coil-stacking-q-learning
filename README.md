# Coil Stacking with Tabular Q-Learning

A reinforcement-learning experiment for rearranging coils in a nine-slot stack. The custom environment encodes support and blocking rules, coil urgency, movement penalties and a stop action. It explores a logistics problem through tabular Q-learning; it does not establish an optimal policy.

## Repository map

- `custom_env.py`: environment used by the training and evaluation scripts.
- `train_model.py`: Q-learning updates and a 500,000-episode training loop; writes `q_table.npy` in the working directory.
- `test_model.py`: bounded evaluation of a trusted saved Q-table on a predefined state.
- `new_case.py`: unfinished alternative environment, with incomplete reward/state handling.

The older README referenced `qlearn.py`, `clean_case.py` and `simple_case.py`; those files are not in the current branch.

## Local setup and smoke check

Use Python 3.10 or 3.11 and the legacy Gym API used by this code:

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'numpy<2' gym==0.26.2
python -m unittest discover -s tests -v
```

The code uses legacy `reset() -> state` and four-value `step()` signatures. It is not a Gymnasium-compatible environment and its declared observation space does not describe the dictionary observations.

For evaluation, run `python test_model.py` only after obtaining or producing a **trusted** Q-table dictionary saved by `train_model.py`. NumPy object arrays use pickle; loading them can execute code. The committed binary was not deserialized during this review. The evaluator ignores invalid actions, stops on unknown states, and has a 100-step safety limit.

To inspect the state/action mechanics without loading a model:

```python
from custom_env import CustomEnv

env = CustomEnv()
state = env.reset()
state, reward, done, info = env.step(81)  # stop action
assert done
```

`python train_model.py` launches the full training loop immediately; inspect its episode count and output behavior before running it. Full training was not run during this review.

## Validation and limitations

Three evaluator regressions passed locally: trusted synthetic table round-trip, unknown-state stop, and valid-action selection with a step limit. An additional smoke call used the real environment and a synthetic empty table. These checks do not measure policy quality.

The state encoder records occupancy but omits priorities, so different urgency assignments can share a Q-table key. The environment increments movement counts in both `step()` and `move_bobbin()`; reward and transition semantics need a separate review. Seeding, train/evaluation separation and baseline comparisons remain open work. No project-wide license is declared.

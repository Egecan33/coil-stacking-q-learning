"""Bounded evaluation of a trusted Q-table saved by train_model.py."""
import numpy as np
from custom_env import CustomEnv


def encode_state(state_dict):
    """Match the occupancy-only encoding used during training."""
    return tuple(state_dict['bobbin_positions'])


def load_q_table(path='q_table.npy'):
    """Load a trusted local artifact. NumPy object files may execute pickle code."""
    table = np.load(path, allow_pickle=True).item()
    if not isinstance(table, dict):
        raise ValueError('Expected a Q-table dictionary saved by train_model.py')
    return table


def test_specific_state(env, q_table, predefined_state, max_steps=100):
    env.set_predefined_state(predefined_state)
    state_dict = env.state
    total_reward = 0
    for step_count in range(1, max_steps + 1):
        state = encode_state(state_dict)
        available = env.find_available_moves()
        q_values = q_table.get(state)
        action = max(available, key=lambda a: q_values[a]) if q_values is not None else 81
        state_dict, reward, done, _ = env.step(action)
        total_reward += reward
        print(f'Step {step_count}: Action taken: {action}')
        env.render()
        print(f'Reward for this step: {reward}, Total Reward: {total_reward}')
        if done:
            return total_reward
    print(f'Evaluation stopped at the {max_steps}-step safety limit.')
    return total_reward


def main():
    q_table = load_q_table()
    env = CustomEnv()
    predefined_state = {
        'bobbin_positions': [1, 1, 0, 0, 1, 0, 0, 0, 0],
        'bobbin_priorities': [2, 2, 0, 0, 10, 0, 0, 0, 0],
    }
    test_specific_state(env, q_table, predefined_state)


if __name__ == '__main__':
    main()

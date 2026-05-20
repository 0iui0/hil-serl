import copy
import os
import select
import signal
import sys
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transitions to collect.")
flags.DEFINE_string("server_url", "http://127.0.0.1:5000/", "URL of the robot server.")
flags.DEFINE_integer("save_every", 20, "Save intermediate data every N successes.")


def save_data(successes, failures, success_needed, suffix=""):
    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"./classifier_data/{FLAGS.exp_name}_{len(successes)}of{success_needed}{suffix}_{ts}"
    s_path = f"{stem}_success.pkl"
    with open(s_path, "wb") as f:
        pkl.dump(successes, f)
    f_path = f"{stem}_failure.pkl"
    with open(f_path, "wb") as f:
        pkl.dump(failures, f)
    print(f"Saved: {len(successes)} success + {len(failures)} failure -> {s_path}")


def main(_):
    print("=" * 60)
    print("CR5AF Reward Classifier Data Collection")
    print("=" * 60)
    print("Controls:")
    print("  s + Enter  -> Mark success for current frame")
    print("  q + Enter  -> Quit and save")
    print("  Ctrl+C     -> Quit and save")
    print("=" * 60)

    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False,
                                 classifier=False, server_url=FLAGS.server_url)

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    save_every = FLAGS.save_every
    pbar = tqdm(total=success_needed)
    mark_success = False

    def on_exit(sig=None, frame=None):
        print(f"\nInterrupted. Saving {len(successes)} success + {len(failures)} failure...")
        save_data(successes, failures, success_needed, suffix="_interrupted")
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)

    while len(successes) < success_needed:
        # Non-blocking stdin check (same pattern as collect_reward_data.py)
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 's':
                mark_success = True
            elif cmd == 'q':
                print("Quit requested.")
                break

        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if mark_success:
            successes.append(transition)
            pbar.update(1)
            mark_success = False
            print(f"\n[s] success #{len(successes)} recorded")
            if len(successes) % save_every == 0:
                save_data(successes, failures, success_needed)
        else:
            failures.append(transition)

        if done or truncated:
            obs, _ = env.reset()

    env.close()
    save_data(successes, failures, success_needed)
    print("Done.")

if __name__ == "__main__":
    app.run(main)

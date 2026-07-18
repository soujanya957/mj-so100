# only-hardware

Keyboard and leader/follower teleop that talks **straight to the real motors**
via [lerobot](https://github.com/huggingface/lerobot). No MuJoCo, no simulation.

## Scripts

| Script | What it does |
|--------|--------------|
| `01-keyboard-teleop-one-robot.py` | Keyboard control of 1 follower arm |
| `02-keyboard-teleop-two-robots.py` | ... 2 follower arms |
| `03-keyboard-teleop-three-robots.py` | ... 3 follower arms |
| `04-keyboard-teleop-four-robots.py` | ... 4 follower arms |
| `05-keyboard-teleop-five-robots.py` | ... 5 follower arms |
| `06-leader-follower-teleop.py` | Move a leader arm, follower mirrors it |
| `so_hw.py` | Shared helpers (imported by the scripts, not run directly) |

## Setup (once)

```bash
pip install lerobot          # or: pip install -e ".[feetech]" from a lerobot checkout
```

### 1. Find each arm's port

```bash
lerobot-find-port
```

Follow the prompt: unplug the arm, press Enter, replug — it prints the port that
disappeared (e.g. `/dev/tty.usbmodem58760431541` on macOS, `COM3` on Windows).
Do this once per arm and note which port is which.

### 2. Set up motors (only for a new arm, or after swapping/replacing a motor)

Each motor needs a unique ID written to it before first use:

```bash
lerobot-setup-motors --port /dev/tty.usbmodemXXXX
```

Plug in motors **one at a time** when prompted. Skip this if the arm already works.

### 3. Calibrate (once per arm)

```bash
lerobot-calibrate --port /dev/tty.usbmodemXXXX --id follower1
```

The `--id` you use here is the calibration name the scripts load with `--id1`,
`--id2`, ... (keyboard) or `--leader-id` / `--follower-id` (leader/follower).

> If your lerobot exposes these as subcommands instead, use
> `python -m lerobot.find_port`, `python -m lerobot.setup_motors`,
> `python -m lerobot.calibrate`.

## Run — keyboard teleop

```bash
# one arm
python 01-keyboard-teleop-one-robot.py --port1 /dev/tty.usbmodemXXXX --id1 follower1

# three arms
python 03-keyboard-teleop-three-robots.py \
    --port1 /dev/tty.usbmodemAAAA --port2 /dev/tty.usbmodemBBBB --port3 /dev/tty.usbmodemCCCC
```

`--idN` defaults to `followerN`, so you only need it if your calibration names differ.

### Keyboard controls

| Key | Action |
|-----|--------|
| `1`..`5` | Select active robot (multi-arm scripts only) |
| `q w e r t y` | Select joint: pan, lift, elbow, wrist-flex, wrist-roll, gripper |
| `↑` / `↓` (or `k` / `j`) | Move selected joint + / − |
| `[` / `]` | Smaller / bigger step |
| `h` / `H` | Home active robot / home all (all joints → 0°) |
| `x` | Quit |

Arms start from their current pose, so nothing jumps on connect. Keep the
terminal focused for keys to register.

## Run — leader / follower

```bash
python 06-leader-follower-teleop.py \
    --leader-port /dev/tty.usbmodemLEAD --follower-port /dev/tty.usbmodemFOLLOW
```

Physically move the leader arm; the follower mirrors it. `Ctrl-C` to stop.
Both arms must be calibrated (default ids `leader` and `follower`; override with
`--leader-id` / `--follower-id`).

## Troubleshooting

- **`ModuleNotFoundError: No module named 'lerobot'`** — install it (see Setup),
  and run from the same Python env you installed it in.
- **Can't tell which port is which** — rerun `lerobot-find-port`, unplugging only
  the arm you're identifying.
- **Permission denied on the port (Linux)** — `sudo usermod -a -G dialout $USER`
  then log out/in, or `sudo chmod 666 /dev/ttyACM0`.
- **macOS port shows but won't open** — it's usually `usbmodem...`, not a `cu.` /
  Bluetooth port; make sure nothing else (another script, a serial monitor) holds it.
- **Motor doesn't move / "id not found"** — motor IDs aren't set: run
  `lerobot-setup-motors` (step 2).
- **Arm drifts or hits limits** — recalibrate with `lerobot-calibrate` for that `--id`.
- **Import error about `SOFollower` / `SO101Follower`** — the scripts try both the
  `so_follower`/`so_leader` and `so101_follower`/`so101_leader` class names. If your
  lerobot uses yet another name, adjust `_load_follower_classes` / `_load_leader_classes`
  in `so_hw.py`.

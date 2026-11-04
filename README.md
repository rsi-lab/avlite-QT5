# avlite-QT5

Qt5 visualizer for [AVLite](https://github.com/AV-Lab/avlite). Same BasicSim workflow as the built-in Tk app (map, vehicle, plans, trajectories, stack controls, profiles), drawn with **pyqtgraph** instead of Matplotlib.

**Registry name:** `avlite-QT5`  
**Category:** `AppStrategy` (`cli_name = "qt5"`)

## Why PyQt5

AVLite supports Python 3.10+. PySide2 has no Python 3.12 wheels. PyQt5 5.15.x publishes `cp38-abi3` wheels and is the Qt5 binding pyqtgraph supports. Plots use `PlotDataItem.setData` (OpenGL when a display is available). Simulation and planning run in `exec.step` / `AsyncThreadedExecuter`, not inside the plot refresh timer.

## Install

```bash
pip install avlite
pip install -r requirements.txt

git clone https://github.com/rsi-lab/avlite-QT5.git \
  ~/.local/share/avlite/plugins/avlite-QT5
```

Register it like any other community plugin (Plugins browser, or the profile snippet below). AVLite imports the directory as `avlite.plugins.avlite_QT5`. That import does not load PyQt.

## Tk visualizer while this plugin is installed

`avlite` builds its Matplotlib figures with `plt.subplots` before it imports community plugins. Once PyQt5 is installed, Matplotlib's automatic backend is `qtagg`, so that call opens a Qt application during the Tk loading screen. Pin the Tk backend in the environment before Python starts:

```bash
MPLBACKEND=TkAgg avlite
```

This repository cannot set that itself. Registered plugins are imported later, from `reload_stack`.

## Launch

The app registers as `qt5`. Once AVLite imports community plugins listed in the profile before it parses subcommands, start it with:

```bash
avlite qt5
avlite qt5 --profile default
```

Stock AVLite 0.6.6 parses subcommands before that import, so on that release use:

```bash
python -m avlite.plugins.avlite_QT5
QT_QPA_PLATFORM=offscreen python -m avlite.plugins.avlite_QT5 --offscreen
```

There is no separate console script in this repo. The core change that loads profile-listed plugins first lives in AVLite, not here.

```yaml
c69_apps:
  c62_community_plugins:
    avlite-QT5: avlite-QT5
```

This repository ships [avlite-QT5.yaml](avlite-QT5.yaml). On Install/Update, AVLite can import that file as a profile named `avlite-QT5`.

Plot mouse matches the Tk visualizer. The pyqtgraph right-click menu is disabled.

- Global plot: left click teleports, left drag sets heading, Ctrl+left drag pans, wheel zooms, right click sets start then goal and replans.
- Local XY / Frenet: left click teleports, right click spawns an NPC, drag sets heading, wheel zooms that pane.
- The Settings bar (profile, Shortcuts / Dark / Global / Local Frenet / Local Global, Save, Reset, Reload Stack, Plugins, gear) sits between the plots and the stack panels.
- The log panel shows live output with Core, Plugins, Perception, Planning, Control, Execution, Visualization, and Common filters, plus level, File, Clear, and Copy.

Shortcut letters match Tk: `S` / `R` / `T` / `E` / `L` / `Q` are the shifted keys. `r` replans, `t` resets execution, `a`/`d` steer until release.

## Layout

Same shape as [avlite-bridge-carla](https://github.com/AV-Lab/avlite-bridge-carla): repository root is the plugin package (`__init__.py`, `settings.py`, strategies, `requirements.txt`, `<name>.yaml`, `README.md`, `test/`).

## Feature-parity checklist (Tk visualizer vs this app, BasicSim)

Headless tests: `QT_QPA_PLATFORM=offscreen python3 -m pytest test/test_feature_parity.py -q`

| Feature | Status |
|---|---|
| Registers `AppStrategy` cli `qt5` via community import | pass |
| BasicSim stack (default profile) | pass |
| Global plot: ego marker + raceline | pass |
| Global plan left/right boundaries | pass |
| Local XY ego bounding box | pass |
| Local Frenet reference path | pass |
| Execution Step advances sim time (SyncExecuter) | pass |
| Start / Stop | pass |
| Reset returns to start pose | pass |
| Teleport ego (plant + stack PM) | pass |
| Spawn NPC agent | pass |
| Switch controller (Stanley) | pass |
| Switch executer (Sync) | pass |
| Perception pipeline refresh | pass |
| Local replan | pass |
| Waypoint step / step back | pass |
| Align plan to ego | pass |
| Align control (snap ego to plan, reset controller) | pass |
| Control step + accel/steer gauges | pass |
| Manual drive (accel/steer, WASD) | pass |
| Save global plan JSON | pass |
| Save occupancy map (no-op when none) | pass |
| Save start pose into the profile | pass |
| Reload with default global plan path | pass |
| World / stack capability filters | pass |
| Probabilistic collision checkbox | pass |
| Shortcut mode hides stack panels | pass |
| Show/hide global and Frenet views | pass |
| Dark / light theme | pass |
| Log panel append + clear | pass |
| LiDAR overlay after a sim step | pass |
| Plot latency recorded and under 500 ms | pass |
| Registers cli `qt5`; `python -m avlite.plugins.avlite_QT5` runs it | pass |

Not identical to Tk (called out so the checklist stays honest):

- The full schema editor, plugin installer, and pip update toast stay in the Tk app (`avlite setting`, `avlite plugins`). This app's gear button edits zoom and the next profile; Plugins lists registered paths.
- HD-map hover of nearest road/lane is drawn when an HD planner is selected; the default BasicSim profile is a race map.
- The local-plot distance ruler, zoom, pan, teleport, and NPC spawn follow the Tk mouse bindings.

## Screenshots

Offscreen renders against BasicSim (Yas Marina):

![Main window](docs/qt_visualizer_main.png)

![Shortcut mode](docs/qt_visualizer_shortcut_mode.png)

## Tests

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest test -q
```

## License

MIT, same as AVLite. PyQt5 is GPL/commercial from Riverbank; use a compatible distribution or a commercial PyQt license if you ship a closed product.

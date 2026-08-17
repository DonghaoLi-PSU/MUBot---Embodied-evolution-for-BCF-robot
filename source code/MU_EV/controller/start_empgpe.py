#!/usr/bin/env python

from MU_EV.controller.mubot_rob_env import MubotEnv # type: ignore
from MU_EV.controller.EM_pgpe import EMPGPE # type: ignore
import numpy as np
import rospy, rospkg
import pandas as pd
import random, os, time, sys, shutil
from pathlib import Path
import threading, re

# ── Rich (optional) ───────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

def _build_panel(morph_id, wk_id,
                 x, n_ep, y, n_ro,
                 rollout_dt, sim_time,
                 mu, act_num,
                 start_time, status=''):
    """Compose the Rich panel shown during PGPE learning."""
    elapsed  = time.time() - start_time
    ep_pct   = (x + 1) / n_ep
    ro_pct   = (y + 1) / n_ro
    bar_w    = 52                        # 66 cols - 2 border - 31 fixed chars

    def pbar(pct, width=bar_w):
        filled = int(pct * width)
        return "█" * filled + "░" * (width - filled)

    # Rollout timing
    overhead     = max(0.0, rollout_dt - sim_time)
    ratio_factor = sim_time / rollout_dt if rollout_dt else float("nan")
    rt_style     = "bright_green" if overhead < 0.5 else ("yellow" if overhead < 2.0 else "red")

    # Freq from μ
    freq_str = f"{mu[-1]:.2f}" if mu is not None and len(mu) else "?"

    t = Text()
    # ── header: Morph | Worker | Act | Elapsed  (fits in 64 chars) ──
    t.append(f"  Morph ", style="dim")
    t.append(str(morph_id), style="bold yellow")
    t.append("  |  Wk ", style="dim")
    t.append(str(wk_id), style="bold magenta")
    t.append(f"  |  Act={act_num}  |  ", style="dim")
    t.append(f"{elapsed:.0f}s\n", style="cyan")

    # ── progress bars ──
    t.append(f"  Epi   ", style="dim")
    t.append(f"{x+1:>3}/{n_ep}  ", style="bold cyan")
    t.append(f"[{pbar(ep_pct)}]\n", style="cyan")

    t.append(f"  Roll  ", style="dim")
    t.append(f"{y+1:>3}/{n_ro}  ", style="bold white")
    t.append(f"[{pbar(ro_pct)}]\n", style="white")

    # ── rollout timing + NYC start time ──
    nyc_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - rollout_dt - 5*3600))
    t.append(f"  Roll  ", style="dim")
    t.append(f"{rollout_dt:.2f}s", style=rt_style)
    t.append(f"  ratio=", style="dim")
    t.append(f"{ratio_factor:.2f}x", style=rt_style)
    t.append(f"  Freq=", style="dim")
    t.append(f"{freq_str}Hz\n", style="bold cyan")
    t.append(f"  Start ", style="dim")
    t.append(f"{nyc_time}  ", style="dim")
    t.append(f"{status}\n" if status else "—\n", style="bright_yellow")

    return Panel(t,
                 title="[bold cyan]EM-PGPE · Controller Optimisation",
                 border_style="cyan",
                 box=box.ROUNDED)


def _nyc_time():
    """Return current time as HH:MM:SS in NYC winter time (UTC-5)."""
    return time.strftime('%H:%M:%S', time.gmtime(time.time() - 5*3600))

def _log(msg, level='W'):
    """Print with NYC winter time, bypassing rospy's formatter."""
    print(f"[{_nyc_time()}] [{level}] {msg}", flush=True)


class _NullCtx:
    """No-op context manager used when Rich is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def update(self, _): pass


class _StatusMonitor:
    """
    Taps fd-1 (stdout) during env.step() via a pipe.
    Parses known strings from Hydro_plugin.cc and roslaunch,
    updates self.status dict which _build_panel reads.
    """
    # Patterns: (compiled_regex, status_key, label_template)
    _PATTERNS = [
        (re.compile(r'urdf_spawner.*process has finished cleanly'), 'spawn',     'URDF spawned'),
        (re.compile(r'Hydro Recording Start'),                      'recording', 'Recording start'),
        (re.compile(r'Hydro Start'),                                'hydro',     'Hydro start'),
        (re.compile(r'Time:\s*([\d.]+)'),                          'simtime',   'Sim t={:.2f}s'),
        (re.compile(r'Hydro Terminate Created'),                    'terminate', 'Terminate flag'),
    ]

    def __init__(self):
        self.status  = {}          # key → (label_str, wall_time)
        self._stop   = False
        self._r_fd   = None
        self._w_fd   = None
        self._saved1 = None
        self._saved2 = None
        self._thread = None

    def start(self):
        """Redirect fd-1/2 to a pipe and start background reader thread."""
        self._r_fd, self._w_fd = os.pipe()
        self._saved1 = os.dup(1)
        self._saved2 = os.dup(2)
        os.dup2(self._w_fd, 1)
        os.dup2(self._w_fd, 2)
        self._stop  = False
        self.status = {}
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        """Restore fd-1/2 and stop the reader thread."""
        os.dup2(self._saved1, 1)
        os.dup2(self._saved2, 2)
        os.close(self._saved1)
        os.close(self._saved2)
        os.close(self._w_fd)   # causes _reader to see EOF
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)
        os.close(self._r_fd)

    def _reader(self):
        buf = b''
        while not self._stop:
            try:
                chunk = os.read(self._r_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b'\n' in buf:
                line_b, buf = buf.split(b'\n', 1)
                self._parse(line_b.decode('utf-8', errors='replace'))

    def _parse(self, line):
        t = time.time()
        for pat, key, tmpl in self._PATTERNS:
            m = pat.search(line)
            if m:
                if '{}' in tmpl and m.lastindex:
                    label = tmpl.format(float(m.group(1)))
                else:
                    label = tmpl
                self.status[key] = (label, t)
                return

    def latest_line(self):
        """Return the most recently seen status label string, or empty string."""
        if not self.status:
            return ''
        key = max(self.status, key=lambda k: self.status[k][1])
        return self.status[key][0]


class MubotPGPE:
    def __init__(self, file_path, max_sim_time, wk_id, morph_ID=None):
        self.env          = MubotEnv(file_path, max_sim_time, wk_id)
        self.file_path    = file_path + "/"
        self.max_sim_time = max_sim_time
        self.morph_ID     = morph_ID if morph_ID is not None else "?"
        if morph_ID is not None:
            self.morph_path = (Path(self.file_path).parent.parent
                               / 'Population_data' / str(morph_ID))
        self._console = Console() if _RICH else None
        self._monitor = _StatusMonitor()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _live_ctx(self, initial_panel=None):
        """Return a Rich Live context (or a no-op if Rich unavailable)."""
        if not _RICH:
            return _NullCtx()
        return Live(initial_panel or Panel("Initialising…"),
                    console=self._console,
                    refresh_per_second=4,
                    screen=True,
                    transient=False)

    def _fallback_log(self, x, n_ep, y, n_ro, rollout_dt):
        print(f"[{_nyc_time()}] [W] Morph={self.morph_ID}  "
              f"E {x+1:>2}/{n_ep}  R {y+1:>2}/{n_ro}  "
              f"t={rollout_dt:.2f}s", flush=True)

    # ── run_pgpe ──────────────────────────────────────────────────────────────

    def run_pgpe(self, wk_id, act_num):
        for fname in ('cpg_history.csv', 'rollout_history.csv', 'reward_history.csv'):
            if os.path.exists(self.file_path + fname):
                os.remove(self.file_path + fname)

        rospack = rospkg.RosPack()
        self.wk_id = wk_id

        if act_num <= 4:
            n_rollout, n_episodes = 40, 30
        elif act_num <= 7:
            n_rollout, n_episodes = 60, 40
        else:
            n_rollout, n_episodes = 80, 50
        # ── Debug override (uncomment to test) ──
        # n_rollout = 10; n_episodes = 2

        n_select = n_rollout // 2
        n_param  = act_num * 2

        record_mu     = np.zeros((n_episodes + 1, n_param))
        record_sigma  = np.zeros((n_episodes + 1, n_param))
        record_reward = np.zeros((n_episodes + 1, n_rollout))
        record_cpg    = np.zeros(((n_episodes + 1) * n_rollout, n_param))

        # Initial μ / σ
        record_mu[0, 0]  = random.uniform(10, 15);  record_sigma[0, 0]  = 5
        for i in range(act_num - 1):
            record_mu[0, 2*i+1] = random.uniform(10, 15); record_sigma[0, 2*i+1] = 5
            record_mu[0, 2*i+2] = random.uniform(0, 1);   record_sigma[0, 2*i+2] = 0.3
        record_mu[0, -1] = random.uniform(0, 10);   record_sigma[0, -1] = 2.5

        empgpe     = EMPGPE(n_param, n_rollout, act_num, n_select)
        start_time = time.time()
        rollout_dt = 0.0

        self._csv_write('cpg_history.csv',
                        np.hstack((record_mu[0, :], record_sigma[0, :])), mode='w')

        with self._live_ctx() as live:
            for x in range(n_episodes):
                record_cpg[x*n_rollout:(x+1)*n_rollout, :] = \
                    empgpe.sample_params(record_mu[x, :], record_sigma[x, :])

                for y in range(n_rollout):
                    temp_input = record_cpg[x * n_rollout + y, :]
                    self._csv_write('control_signal.csv', temp_input, mode='w')

                    t0 = time.time()
                    self._monitor.start()
                    record_reward[x, y] = self.env.step()
                    self._monitor.stop()
                    rollout_dt = time.time() - t0

                    if _RICH:
                        live.update(_build_panel(
                            self.morph_ID, wk_id,
                            x, n_episodes, y, n_rollout,
                            rollout_dt, self.max_sim_time,
                            record_mu[x, :],
                            act_num, start_time,
                            self._monitor.latest_line()))
                    else:
                        self._fallback_log(x, n_episodes, y, n_rollout, rollout_dt)

                    mode = 'w' if (x == 0 and y == 0) else 'a'
                    self._csv_write('rollout_history.csv',
                                    np.hstack((temp_input, record_reward[x, y])), mode=mode)

                self.env._cleanup_ros_logs()

                record_mu[x+1, :], record_sigma[x+1, :], _ = empgpe.learn(
                    record_cpg[x*n_rollout:(x+1)*n_rollout, :],
                    record_reward[x, :],
                    record_mu[x, :], record_sigma[x, :])

                self._csv_write('cpg_history.csv',
                                np.hstack((record_mu[x+1, :], record_sigma[x+1, :])))
                mode = 'w' if x == 0 else 'a'
                self._csv_write('reward_history.csv', record_reward[x, :], mode=mode)

                mean_r = np.mean(record_reward[x, :])
                std_r  = np.std(record_reward[x, :])
                if std_r <= mean_r * 0.1 and x >= n_episodes * 0.5:
                    _log(f"[PGPE] Early converge at episode {x+1}", level='E')
                    break

        final_reward = np.mean(record_reward[x, :])
        final_cpg    = np.hstack((record_mu[x+1, :], record_sigma[x+1, :]))
        if self.morph_ID != "?" and os.path.exists(self.file_path + 'gait_record.csv'):
            shutil.move(self.file_path + 'gait_record.csv', self.morph_path / 'gait_.csv')
        return final_reward, final_cpg

    # ── continue_pgpe ─────────────────────────────────────────────────────────

    def continue_pgpe(self, wk_id, act_num):
        rospack = rospkg.RosPack()
        self.wk_id = wk_id

        if act_num <= 4:
            n_rollout, n_episodes = 40, 35
        elif act_num <= 7:
            n_rollout, n_episodes = 60, 45
        else:
            n_rollout, n_episodes = 80, 55

        n_select = n_rollout // 2
        n_param  = act_num * 2

        record_mu     = np.zeros((n_episodes + 1, n_param))
        record_sigma  = np.zeros((n_episodes + 1, n_param))
        record_reward = np.zeros((n_episodes + 1, n_rollout))
        record_cpg    = np.zeros(((n_episodes + 1) * n_rollout, n_param))

        # Restore history
        cpg_hist    = pd.read_csv(self.file_path+'cpg_history.csv',    header=None).to_numpy()
        ep_comp     = cpg_hist.shape[0] - 1
        reward_hist = pd.read_csv(self.file_path+'reward_history.csv', header=None).to_numpy()[:ep_comp, :]
        pd.DataFrame(reward_hist).to_csv(self.file_path+'reward_history.csv',
                                         float_format='%.8f', header=None, index=None, mode='w')
        rollout_hist = pd.read_csv(self.file_path+'rollout_history.csv', header=None).to_numpy()[:ep_comp*n_rollout, :]
        pd.DataFrame(rollout_hist).to_csv(self.file_path+'rollout_history.csv',
                                          float_format='%.8f', header=None, index=None, mode='w')

        record_mu[:ep_comp+1, :]    = cpg_hist[:ep_comp+1, :n_param]
        record_sigma[:ep_comp+1, :] = cpg_hist[:ep_comp+1, n_param:]
        record_reward[:ep_comp, :]  = reward_hist
        record_cpg[:rollout_hist.shape[0], :] = rollout_hist[:rollout_hist.shape[0], :-1]

        empgpe     = EMPGPE(n_param, n_rollout, act_num, n_select)
        start_time  = time.time()
        rollout_dt  = 0.0

        if ep_comp == n_episodes - 1:
            x = ep_comp

        with self._live_ctx() as live:
            for x in range(ep_comp, n_episodes):
                record_cpg[x*n_rollout:(x+1)*n_rollout, :] = \
                    empgpe.sample_params(record_mu[x, :], record_sigma[x, :])

                for y in range(n_rollout):
                    temp_input = record_cpg[x * n_rollout + y, :]
                    self._csv_write('control_signal.csv', temp_input, mode='w')

                    t0 = time.time()
                    self._monitor.start()
                    record_reward[x, y] = self.env.step()
                    self._monitor.stop()
                    rollout_dt = time.time() - t0

                    if _RICH:
                        live.update(_build_panel(
                            self.morph_ID, wk_id,
                            x, n_episodes, y, n_rollout,
                            rollout_dt, self.max_sim_time,
                            record_mu[x, :],
                            act_num, start_time,
                            self._monitor.latest_line()))
                    else:
                        self._fallback_log(x, n_episodes, y, n_rollout, rollout_dt)

                    mode = 'w' if (x == ep_comp and y == 0) else 'a'
                    self._csv_write('rollout_history.csv',
                                    np.hstack((temp_input, record_reward[x, y])), mode=mode)

                record_mu[x+1, :], record_sigma[x+1, :], _ = empgpe.learn(
                    record_cpg[x*n_rollout:(x+1)*n_rollout, :],
                    record_reward[x, :],
                    record_mu[x, :], record_sigma[x, :])

                self._csv_write('cpg_history.csv',
                                np.hstack((record_mu[x+1, :], record_sigma[x+1, :])))
                mode = 'w' if x == ep_comp else 'a'
                self._csv_write('reward_history.csv', record_reward[x, :], mode=mode)

                mean_r = np.mean(record_reward[x, :])
                std_r  = np.std(record_reward[x, :])
                if std_r <= mean_r * 0.1:
                    _log(f"[PGPE] Early converge at episode {x+1}", level='E')
                    break

        final_reward = np.mean(record_reward[x, :])
        final_cpg    = np.hstack((record_mu[x, :], record_sigma[x, :]))
        print("Final cpg: " + str(final_cpg))
        return final_reward, final_cpg

    # ── one_pepg ──────────────────────────────────────────────────────────────

    def one_pepg(self, wk_id, morph_ID, act_num):
        rospack = rospkg.RosPack()
        self.wk_id = wk_id

        n_param    = act_num * 2
        reward     = 0.0

        _log(f"Rerun for morph ID: {morph_ID}")
        df         = pd.read_csv(self.file_path + '/controller.csv',
                                 header=None, index_col=None).to_numpy().flatten()
        temp_input = df[:-1]
        self._csv_write('control_signal.csv', temp_input, mode='w')

        t0     = time.time()
        reward = self.env.step()
        dt     = time.time() - t0

        _log(f"Reward: {reward:.6e}  t={dt:.2f}s", level='E')
        return reward

    # ── _csv_write ────────────────────────────────────────────────────────────

    def _csv_write(self, filename, data, mode='a'):
        """Write CSV and guarantee the data is flushed to disk before returning.
        Prevents Gazebo plugin from reading a partially-written file."""
        df_data  = pd.DataFrame(data.reshape(1, -1))
        filepath = self.file_path + filename
        with open(filepath, mode, newline='') as f:
            df_data.to_csv(f, float_format='%.8f', header=None, index=None)
            f.flush()
            os.fsync(f.fileno())

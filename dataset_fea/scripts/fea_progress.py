"""Read Abaqus Explicit status tails and display per-sample simulation progress."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import threading
from tqdm import tqdm


NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[ED][+-]?\d+)?'
ROW = re.compile(r'^\s*(\d+)\s+('+NUMBER+r')\s+('+NUMBER+r')\s+\d+:\d{2}:\d{2}\s+('+NUMBER+r')\s+\d+(?:\s|$)', re.I)
SUCCESS = 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY'


@dataclass
class Status:
    step_time: float = 0.0
    increment: int = 0
    completed: bool = False


def read_status(path):
    """Read at most 32 KiB regardless of how large the solver log becomes."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2); offset = max(0, stream.tell()-32768); stream.seek(offset)
            lines = stream.read().decode('ascii', errors='replace').splitlines()
        if offset: lines = lines[1:]
    except OSError:
        return Status()
    completed = any(SUCCESS in line.upper() for line in lines)
    for line in reversed(lines):
        match = ROW.match(line)
        if match:
            step = float(match[2].upper().replace('D','E'))
            if math.isfinite(step) and step >= 0:
                return Status(step, int(match[1]), completed)
    return Status(completed=completed)


class SampleProgress:
    def __init__(self, case, sample_id, total_samples, step_seconds=None, interval=1.0, file=None, disable=None):
        self.case = Path(case); self.sample_id = sample_id; self.total_samples = total_samples
        if step_seconds is None:
            analysis = json.loads((self.case/'analysis.json').read_text(encoding='utf-8'))
            step_seconds = analysis.get('step_time_s', 1.0)
        if not math.isfinite(step_seconds) or step_seconds <= 0: raise ValueError('Positive step duration required')
        self.step_seconds = step_seconds; self.interval = interval
        self.file = file or sys.stderr; self.disable = disable
        self.stop = threading.Event()

    def percentage(self, status):
        if status.completed: return 100.0
        return min(99.99, max(0.0, status.step_time/self.step_seconds*100))

    def tick(self):
        status = read_status(self.case/'analysis.sta')
        percentage = max(self.bar.n, self.percentage(status))
        try:
            stage = json.loads((self.case/'state.json').read_text(encoding='utf-8'))['stage']
        except (OSError, ValueError, KeyError): stage = 'WAITING'
        self.bar.set_postfix(time='%.5g/%gs' % (status.step_time,self.step_seconds),
            inc=status.increment, stage=stage, refresh=False)
        self.bar.update(percentage-self.bar.n)
        self.bar.refresh()

    def monitor(self):
        while not self.stop.wait(self.interval): self.tick()

    def __enter__(self):
        initial = self.percentage(read_status(self.case/'analysis.sta'))
        self.bar = tqdm(total=100.0, initial=initial,
            desc='Sample %d (%d/%d)' % (self.sample_id,self.sample_id+1,self.total_samples),
            file=self.file, disable=self.disable, dynamic_ncols=True, ascii=True,
            mininterval=self.interval, leave=True,
            bar_format='{desc}: {percentage:6.2f}%|{bar}| [{elapsed}<{remaining}, {postfix}]')
        self.tick()
        self.thread = threading.Thread(target=self.monitor, name='fea-progress', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set(); self.thread.join(timeout=3)
        self.tick()
        if exc_type is not None:
            self.bar.set_postfix_str('paused' if issubclass(exc_type, KeyboardInterrupt) else 'interrupted')
        self.bar.close()
        return False

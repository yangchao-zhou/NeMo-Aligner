import os
import signal
import psutil

for proc in psutil.process_iter(['pid', 'name']):
    if proc.info['name'] == 'python':
        os.kill(proc.info['pid'], signal.SIGKILL)

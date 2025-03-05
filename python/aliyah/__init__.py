from contextlib import contextmanager
from .monitor import monitor
__version__ = "0.1.0"

@contextmanager 
def trainingmonitor():
    """
    Context manager for monitoring ML trainign 
    """
    print("__ALIYAH_MONITOR_START__")
    try:
        yield monitor
    finally:
        monitor.should_stop = True
        print("__ALIYAH_MONITOR_END__")

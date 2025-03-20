from contextlib import contextmanager
from .monitor import monitor
__version__ = "0.1.0"

@contextmanager 
def trainingmonitor():
    """
    Wrapper for hooks and for monitoring ML training 
    """
    print("__ALIYAH_MONITOR_START__")
    try:
        yield monitor
    finally:
        monitor.should_stop = True
        print("__ALIYAH_MONITOR_END__")

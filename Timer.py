import time


class Timer:
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.msg:
            print(f"{self.msg} took {self.interval:0.3f} sec")

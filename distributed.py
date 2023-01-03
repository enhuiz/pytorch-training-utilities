import os
import socket
from functools import cache, wraps
from typing import Callable

import deepspeed


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


@cache
def init_distributed():
    deepspeed.init_distributed()


def local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def global_rank():
    return int(os.environ.get("RANK", 0))


def is_local_leader():
    return local_rank() == 0


def is_global_leader():
    return global_rank() == 0


def local_leader_only(fn=None, *, default=None):
    def wrapper(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if is_local_leader():
                return fn(*args, **kwargs)
            return default

        return wrapped

    if fn is None:
        return wrapper

    return wrapper(fn)


def global_leader_only(fn: Callable | None = None, *, default=None):
    def wrapper(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if is_global_leader():
                return fn(*args, **kwargs)
            return default

        return wrapped

    if fn is None:
        return wrapper

    return wrapper(fn)


def nondistributed(fn):
    @global_leader_only()
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # https://github.com/microsoft/DeepSpeed/blob/b47e25bf95250a863edb2c466200c697e15178fd/deepspeed/utils/distributed.py#L34
        # Deepspeed will check all environ before start distributed.
        # To avoid the start of a distributed task, remove one environ is enough.
        # Here we remove local rank.
        local_rank = os.environ.pop("LOCAL_RANK", "")
        ret = fn(*args, **kwargs)
        os.environ["LOCAL_RANK"] = local_rank
        return ret

    return wrapped

from mmengine.config import read_base

with read_base():
    from .TCMbench_c1 import * # noqa: F401, F403

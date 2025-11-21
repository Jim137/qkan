# MIT License

# Copyright (c) 2025 Andrej Karpathy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Helpful info functions for QKAN.
Parts of codes are adapted from https://github.com/karpathy/nanochat.
"""

import os

from . import __version__


def print0(*s, **kwargs):
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(*s, **kwargs)


def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
        ██████    █████   ████   █████████   ██████   █████
      ███░░░░███ ░░███   ███░   ███░░░░░███ ░░██████ ░░███ 
     ███    ░░███ ░███  ███    ░███    ░███  ░███░███ ░███ 
    ░███     ░███ ░███████     ░███████████  ░███░░███░███ 
    ░███   ██░███ ░███░░███    ░███░░░░░███  ░███ ░░██████ 
    ░░███ ░░████  ░███ ░░███   ░███    ░███  ░███  ░░█████ 
     ░░░██████░██ █████ ░░████ █████   █████ █████  ░░█████
       ░░░░░░ ░░ ░░░░░   ░░░░ ░░░░░   ░░░░░ ░░░░░    ░░░░░     
    """
    print0(banner)


def print_version():
    # print the version and the banner
    print0("=" * 60)
    print_banner()
    print0(f"QKAN version: {__version__}")
    print0("=" * 60)


def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get("RANK", -1)) != -1


def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

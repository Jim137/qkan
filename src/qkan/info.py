import os

from . import __version__


def print0(*s, **kwargs):
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(s, **kwargs)


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
    # print the version and the banner"
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

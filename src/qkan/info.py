import os

from . import __version__


def print0(s="", **kwargs):
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

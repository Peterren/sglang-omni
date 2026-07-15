# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the qwen3-text-server preset."""

try:
    from examples import _omni_launcher as _launcher
except ModuleNotFoundError:
    import _omni_launcher as _launcher

_launch_text_server = _launcher.launch_qwen_text_server


def parse_args():
    return _launcher.parse_preset_args("qwen3-text-server")


def main() -> None:
    _launcher.run_preset("qwen3-text-server")


if __name__ == "__main__":
    main()

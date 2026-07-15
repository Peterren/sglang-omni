# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the ming-speech-server preset."""

try:
    from examples import _omni_launcher as _launcher
except ModuleNotFoundError:
    import _omni_launcher as _launcher


def _launch_speech_server(args):
    return _launcher.launch_ming_speech_server(args)


def parse_args():
    return _launcher.parse_preset_args("ming-speech-server")


def main() -> None:
    _launcher.run_preset("ming-speech-server")


if __name__ == "__main__":
    main()

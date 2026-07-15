# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the qwen3-speech-server preset."""

try:
    from examples import _omni_launcher as _launcher
except ModuleNotFoundError:
    import _omni_launcher as _launcher


def _launch_speech_server(args):
    return _launcher.launch_qwen_speech_server(args)


def _parse_thinker_tp_gpu_list(spec: str, tp_size: int) -> list[int]:
    return _launcher._parse_thinker_tp_gpu_list(spec, tp_size)


def parse_args():
    return _launcher.parse_preset_args("qwen3-speech-server")


def main() -> None:
    _launcher.run_preset("qwen3-speech-server")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
bittools — entry point

Usage:
    python main.py              # Launch main bit viewer (bitabyte)
    python main.py --nextgen    # Launch next-gen viewer prototype
    python main.py --generate   # Run interactive frame data generator (CLI)
    python main.py --process    # Run bit processing CLI tool
"""

import sys


def _launch_qt_window(window_class):
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = window_class()
    window.show()
    sys.exit(app.exec())


def launch_viewer():
    from viewer.window import BitViewerWindow
    _launch_qt_window(BitViewerWindow)


def launch_nextgen():
    from nextgen.main_window import NextGenBitViewerWindow
    _launch_qt_window(NextGenBitViewerWindow)


def launch_generator():
    from generator.generator import FrameDataGenerator
    gen = FrameDataGenerator()
    gen.interactive_session()


def launch_processing():
    from processing.cli import BitProcessorCLI
    cli = BitProcessorCLI()
    cli.run()


if __name__ == "__main__":
    if "--nextgen" in sys.argv:
        launch_nextgen()
    elif "--generate" in sys.argv:
        launch_generator()
    elif "--process" in sys.argv:
        launch_processing()
    else:
        launch_viewer()

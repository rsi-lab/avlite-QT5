"""Entry point: ``python -m avlite.plugins.avlite_QT5``.

``avlite qt5`` is the normal launch once AVLite imports profile-listed
community plugins before it parses subcommands. This module stays free of
Qt imports so that scan can run during startup.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    import avlite.plugins  # registers the community import hook
    import avlite.plugins.avlite_QT5  # noqa: F401 — registers VisualizationQtApp

    from avlite.c60_apps.c61_app_strategy import AppStrategy

    cls = AppStrategy.registry.get("qt5")
    if cls is None:
        print("Qt visualizer AppStrategy did not register.", file=sys.stderr)
        return 1
    parser = argparse.ArgumentParser(prog="python -m avlite.plugins.avlite_QT5")
    cls().configure_parser(parser)
    args, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        args.offscreen = True
    code = cls().run(args, unknown)
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())

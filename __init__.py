"""AVLite Qt5 visualizer community plugin.

AVLite loads this directory as ``avlite.plugins.avlite_QT5`` when it is installed
under the community plugin path (``~/.local/share/avlite/plugins/avlite-QT5``)
or listed in ``c62_community_plugins``. Importing the package registers
``VisualizationQtApp`` (``cli_name = "qt"``).

Stock AVLite only auto-imports built-in ``p60_*`` packages before parsing
``avlite <command>``. Launch this app with::

    python -m avlite.plugins.avlite_QT5
"""

if __package__:
    from .app_strategy import VisualizationQtApp
    from .settings import PluginSettings, PluginSettingsSchema

    __all__ = [
        "VisualizationQtApp",
        "PluginSettings",
        "PluginSettingsSchema",
    ]
else:  # pragma: no cover
    __all__ = []

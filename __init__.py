"""AVLite Qt5 visualizer community plugin.

AVLite loads this directory as ``avlite.plugins.avlite_QT5`` when it is listed
in ``c62_community_plugins`` (or installed under the community plugin path).
Importing the package registers ``VisualizationQtApp`` (``cli_name = "qt5"``).

This import does not load PyQt. The window is created only when the app runs,
so AVLite can import the plugin while it is still parsing ``avlite qt5``.
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

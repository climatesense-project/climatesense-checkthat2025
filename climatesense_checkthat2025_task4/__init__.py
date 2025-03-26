from platformdirs import user_data_path
from usingversion import getattr_with_version

APPNAME = "climatesense-checkthat2025-task4"
APP_REPOSITORY = "https://github.com/climatesense-project/climatesense-checkthat2025-task4"
DEFAULT_ROOT_PATH = user_data_path(appname=APPNAME)

__getattr__ = getattr_with_version(APPNAME, __file__, __name__)

import os
import sys
import base64

from loguru import logger

cur_dir = os.path.dirname(__file__)
root_dir = os.path.join(cur_dir, "..")


def get_path(cfg_file_path):
    """
    Get actual path for file in config
    - Param:
      + cfg_file_path: file path, which need to read from config file

    - Return:
      + path to the file specified in config file
    """
    file_path = os.path.join(root_dir, cfg_file_path)
    if os.path.isfile(file_path):
        return file_path
    else:
        logger.error("File {} does not exist".format(file_path))
        sys.exit(0)


def isBase64(sb):
    """Check String is base64 or not

    Args:
        sb(str): Input string
    """
    try:
        if isinstance(sb, str):
            # If there's any unicode here, an exception will be thrown and the function will return false
            sb_bytes = bytes(sb, "ascii")
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
        return False

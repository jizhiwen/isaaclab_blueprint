import os

ISAACLAB_BLUEPRINT_LOCAL_DIR = os.environ["ISAACLAB_BLUEPRINT_ASSET_ROOT_DIR"] \
    if "ISAACLAB_BLUEPRINT_ASSET_ROOT_DIR" in os.environ \
    else f'{os.environ["HOME"]}/isaaclab_blueprint_assets'
"""Path to the isaaclab_bleuprint assets root directory on local."""

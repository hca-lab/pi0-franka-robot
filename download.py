import os
from openpi.shared import download

# Hard-code your preferred local storage path
os.environ["OPENPI_DATA_HOME"] = "/home/yi/ModelCheckPoints/VLA-Model"

# Trigger the download
local_path = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
print("Checkpoint stored at:", local_path)

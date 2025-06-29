"""locate test data"""

import os
from pathlib import Path


_this_file_path = Path(os.path.abspath(__file__))
_test_data_dir = _this_file_path.parent
dem_path = _test_data_dir / "dem.tif"

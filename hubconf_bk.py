# Wrapper hubconf.py
# Chooses between patched and original implementations at import time.
# Usage:
#   set Y5_PATCHED=1  (default) -> use patched
#   set Y5_PATCHED=0            -> use original

import os
if os.getenv("Y5_PATCHED", "1") == "1":
    from hubconf_patched import *  # noqa
else:
    from hubconf_orig import *     # noqa

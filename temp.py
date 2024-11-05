import numpy as np
import re
ss = "mathew <mathew@mantis.co.uk>"
print(re.sub(r"[^a-zA-Z0-9]", " ", ss))

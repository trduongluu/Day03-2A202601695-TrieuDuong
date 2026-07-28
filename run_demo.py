import os
import sys

# Explicitly add site-packages paths containing streamlit
roaming_site = r"C:\Users\ADMIN\AppData\Roaming\Python\Python312\site-packages"
if roaming_site not in sys.path:
    sys.path.insert(0, roaming_site)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.port", "8501"]
    sys.exit(stcli.main())

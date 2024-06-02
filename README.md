# Intra Cluster Filtering

## Installation Instructions

We recommend creating an environment for the installation of the library.

```sh
# Create a new environment
python -m venv env_name

# Activate the environment
# On Windows
.\env_name\Scripts\activate

# On macOS/Linux
source env_name/bin/activate

# Install the package
pip install -r requirements.txt
python -m build
pip install dist/intraclusterfiltering-0.0.1-py3-none-any.whl

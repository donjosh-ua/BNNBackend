import json
import re
import os

class Config:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(__file__), "settings.conf")
        self.settings = {}
        self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                content = f.read().strip()
                if content:
                    # Parse the setting manually since it's not standard JSON
                    match = re.search(r'"data_file":\s*"([^"]+)"', content)
                    if match:
                        self.settings["data_file"] = match.group(1)
        except FileNotFoundError:
            # Use default value if file doesn't exist
            self.settings["data_file"] = "./app/data/pima-indians-diabetes.data.csv"
    
    def get(self, key, default=None):
        return self.settings.get(key, default)

# Create a singleton instance
config = Config()

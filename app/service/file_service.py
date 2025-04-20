import os
import json
import re
from typing import Dict, List, Tuple

class FileService:
    """Service for managing files in the application"""
    
    def __init__(self):
        self.data_dir = "./app/data"
        self.config_file = "./app/config/settings.conf"
    
    def get_available_files(self) -> Dict[str, List[str]]:
        """
        Get all available data files categorized by type (csv, image)
        
        Returns:
            Dict with two keys: 'csv' and 'image', each containing a list of file paths
        """
        files = {
            "csv": [],
            "image": []
        }
        
        # Walk through the data directory
        for root, dirs, filenames in os.walk(self.data_dir):
            for filename in filenames:
                # Skip __init__.py and other non-data files
                if filename.startswith('__') or filename.startswith('.'):
                    continue
                    
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, start=os.path.dirname(self.data_dir))
                
                # Categorize by extension
                if filename.lower().endswith('.csv'):
                    files["csv"].append(rel_path)
                elif any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                    files["image"].append(rel_path)
        
        return files
    
    def set_selected_dataset(self, file_path: str) -> Dict[str, str]:
        """
        Update the settings.conf file with the selected dataset
        
        Args:
            file_path: Path to the selected data file
            
        Returns:
            Dict with the update status
        """
        # Validate file exists
        full_path = os.path.join("./app", file_path) if not file_path.startswith('./app') else file_path
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Update settings
        with open(self.config_file, 'w') as f:
            f.write(f'"data_file": "{full_path}",\n')
        
        return {"message": "Dataset selected successfully", "file": file_path}
    
    def get_current_dataset(self) -> Dict[str, str]:
        """
        Get the currently selected dataset from settings.conf
        
        Returns:
            Dict with the current dataset file path
        """
        try:
            with open(self.config_file, 'r') as f:
                content = f.read().strip()
                if content:
                    # Parse the setting manually since it's not standard JSON
                    match = re.search(r'"data_file":\s*"([^"]+)"', content)
                    if match:
                        return {"data_file": match.group(1)}
                return {"data_file": ""}
        except FileNotFoundError:
            return {"data_file": ""} 
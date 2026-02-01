import os
import zipfile
import shutil  
zip_file_path="archive.zip"
des_file_path="data"
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(des_file_path)

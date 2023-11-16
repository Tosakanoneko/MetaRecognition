import os
import shutil

def clear_directory(dir_path):
    """
    Check if the directory at dir_path is empty. If not, clear its contents.
    
    :param dir_path: Path to the directory to be checked and cleared.
    """
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' does not exist.")
        return
    
    if not os.path.isdir(dir_path):
        print(f"'{dir_path}' is not a directory.")
        return
    
    # Check if directory is empty
    if os.listdir(dir_path):
        print(f"Directory '{dir_path}' is not empty. Clearing contents...")
        
        # Clear the contents of the directory
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Directory '{dir_path}' is already empty.")

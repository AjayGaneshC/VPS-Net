import os
import subprocess
import sys

def compile_cuda_extension():
    # Path to the setup.py script
    setup_script_path = 'lib/module/PNS/setup.py'
    
    # Ensure you're in the correct directory
    project_root = os.path.dirname(os.path.abspath(setup_script_path))
    
    try:
        # Change to the directory containing setup.py
        os.chdir(project_root)
        
        # Run the setup script to compile the extension
        result = subprocess.run([
            sys.executable, 
            'setup.py', 
            'build_ext', 
            '--inplace'
        ], check=True, capture_output=True, text=True)
        
        print("CUDA extension compiled successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error compiling CUDA extension:")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    compile_cuda_extension()

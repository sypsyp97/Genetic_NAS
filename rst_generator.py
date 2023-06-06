# Import the os module, which provides functions for interacting with the operating system.
import os

# Specify the directory where you want to create the .rst files.
rst_directory = "docs"

# Create the directory if it doesn't exist.
os.makedirs(rst_directory, exist_ok=True)

# Specify the directory you want to scan for python files.
py_directory = "src"

# Get a list of filenames in the directory you specified.
files = os.listdir(py_directory)

# Filter out non-python files.
py_files = [f for f in files if f.endswith(".py")]

rst_files = []
for file in py_files:
    # Remove the .py extension to get the module name.
    module_name = file[:-3]
    rst_files.append(module_name)

    # Create a new .rst file with the same name.
    rst_file = f"{rst_directory}/{module_name}.rst"

    # Open the .rst file in write mode.
    with open(rst_file, "w") as f:
        # Write the content to the .rst file.
        f.write(
            f"""{module_name}
{'=' * len(module_name)}

.. automodule:: {py_directory}.{module_name}
   :members:
"""
        )

# Print a message to the console indicating that the .rst files have been generated.
print("All .rst files have been generated.")

# Update the index.rst file.
index_rst = f"{rst_directory}/index.rst"

# Open the index.rst file in append mode.
with open(index_rst, "a") as f:
    for rst_file in rst_files:
        # Append the name of each .rst file to the index.rst file.
        f.write(f"   {rst_file}\n")

# Print a message to the console indicating that the index.rst file has been updated.
print("index.rst has been updated.")

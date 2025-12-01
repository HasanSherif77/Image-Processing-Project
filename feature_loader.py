# feature_loader.py
import json
import types


def load_feature(ipynb_path, function_name="apply"):
    """
    Load a function from a .ipynb file as if it were a .py module.
    The notebook MUST contain a function with the given name.
    """

    # Read the notebook
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Extract Python code from code cells
    code_cells = [
        "".join(cell["source"]) for cell in nb["cells"]
        if cell["cell_type"] == "code"
    ]

    # Combine into one executable script
    full_code = "\n\n".join(code_cells)

    # Create a temporary module
    module = types.ModuleType("temp_feature_module")

    # Execute code inside module's namespace
    exec(full_code, module.__dict__)

    # Validate presence of the apply() function
    if function_name not in module.__dict__:
        raise ValueError(
            f"Function '{function_name}' not found in {ipynb_path}. "
            "Make sure the notebook defines it."
        )

    return module.__dict__[function_name]

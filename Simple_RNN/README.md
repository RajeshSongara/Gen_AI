# Simple RNN Project

This folder contains the Simple RNN notebook and its Python dependencies.

## Run on Windows PowerShell

From the repository root, run:

```powershell
cd simple_rnn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requiremnet.txt
jupyter notebook embedding.ipynb
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run in VS Code

1. Open `embedding.ipynb`.
2. Select the Python interpreter from `simple_rnn/.venv`.
3. Install the packages from `requiremnet.txt` if VS Code asks.
4. Run the notebook cells from top to bottom.

The notebook creates a Keras embedding model from the sample sentences.
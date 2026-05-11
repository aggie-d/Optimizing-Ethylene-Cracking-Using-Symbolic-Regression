# Optimizing Ethylene Cracking with Symbolic Regression

This project uses symbolic regression to discover interpretable equations for classifying ethylene cracking operating conditions as feasible or infeasible. The main machine-learning pipeline is in `Regression.py`: it loads the process dataset, splits it into train/validation/test sets, trains a PySR model, evaluates the saved equation, and stores trial metadata in a SQLite database.

The practical goal is to produce a compact feasibility-boundary equation, not just a black-box classifier.

**IMPORTANT:**
The dataset used for this project has been omitted from this repo. The dataset must included all variables used. The dataset must be renamed to `AllSAMPLES.xlsx`, or the code must be updated to reflect the new name of the dataset. 

## Project Overview

The pipeline works with reactor/process variables from `AllSAMPLES.xlsx`. `Regression.py` renames the spreadsheet columns to:

- `Tin`
- `q`
- `flow_shale`
- `flow_steam`
- `length`
- `Pressure`
- `Status`
- `Feasability`

`Feasability` is the target label. The current model uses these input variables by default:

```python
["Tin", "q", "flow_shale", "flow_steam", "length", "Pressure"]
```

The project treats PySR output as a binary classifier by taking the sign of the symbolic-regression score:

- positive output: feasible
- negative output: infeasible
- zero output: treated as infeasible

## Repository Contents

| Path | Purpose |
| --- | --- |
| `Regression.py` | Main ML pipeline: load/split data, train PySR, validate, test, plot metrics, and save trial results. |
| `extract.py` | Exports top model records from `Results.db` to `Top_10_PySR_Models.csv`. |
| `figure.py` | Creates a manually specified confusion-matrix figure. |
| `Opti_Figure.py` | Generates `methodology_optimization_viz.png`, a visual explanation of the symbolic-regression workflow. |
| `feasable.py` | Simple standalone feasibility-boundary visualization. |
| `requirements.txt` | Pinned Python package requirements for the project environment. |
| `Results.db` | SQLite database where pipeline trials are stored. |
| `Top_10_PySR_Models.csv` | CSV export of selected top model records. |
| `AllSAMPLES.xlsx` | Input spreadsheet expected by the main pipeline. |
| `my_equations/` | PySR output directory containing saved runs and hall-of-fame files. |

## Requirements

This project uses Python plus PySR. All Python package requirements are listed in `requirements.txt`. PySR also uses Julia through its Python integration, so the first PySR run on a new machine may take extra time while Julia dependencies are installed.

Install the Python dependencies with:

```powershell
pip install -r requirements.txt
```

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Make sure `AllSAMPLES.xlsx` is in the project root before running the pipeline.

## Running the ML Pipeline

Run the full pipeline with:

```powershell
python Regression.py
```

The current `main()` function sets:

```python
path = "my_equations/2_16_26.0.5"
run_id = "2_16_26.0.5"
num_repeat = 1
```

Then it calls:

```python
start(all_variables, run_id, path, num_repeat)
```

For each trial, the pipeline:

1. Loads `AllSAMPLES.xlsx`.
2. Randomly selects a train/validation/test split.
3. Trains a PySR model with `Training_Set(...)`.
4. Loads the saved PySR run from disk for validation.
5. Loads the saved PySR run again for testing.
6. Prints precision, recall, F1, AUC, and accuracy.
7. Displays a confusion matrix.
8. Saves trial metadata to `Results.db`.

## Training Configuration

The PySR model is configured in `Training_Set(...)` with:

- `niterations=1000`
- `populations=50`
- `maxsize=50`
- binary operators: `+`, `*`, `-`, `/`
- unary operators: `exp`, `square`, `sqrt`, `inv`, `cube`, `log`
- `elementwise_loss='SigmoidLoss()'`
- `model_selection="accuracy"`
- output directory: `my_equations`

To run a different experiment, edit these values in `main()`:

```python
path = "my_equations/<experiment_name>"
run_id = "<experiment_name>"
num_repeat = 1
```

If `num_repeat` is greater than `1`, `start()` modifies the end of `path` and `run_id` for each repeated trial. For repeated experiments, it is easiest to use names ending in a period, such as:

```python
path = "my_equations/3_8_26.0."
run_id = "3_8_26.0."
num_repeat = 5
```

This creates runs ending in `0`, `1`, `2`, and so on.

## Saved Results

`print_results(...)` writes each trial to the `results` table in `Results.db` with:

- trial ID
- test accuracy
- validation accuracy
- variables used
- random state
- runtime
- model complexity
- LaTeX equation

The model files themselves are saved by PySR under `my_equations/<run_id>/`.

## Exporting Top Models

Run:

```powershell
python extract.py
```

This reads from `Results.db` and writes:

```text
Top_10_PySR_Models.csv
```

If the database schema changes, update the SQL query in `extract.py` so it matches the columns actually stored by `print_results(...)`.

## Comparing Equation Structure

`Regression.py` also includes helper functions for comparing symbolic equations:

- `compare_pysr_runs(file1, file2, top_n=5)` compares top equations from two PySR hall-of-fame CSV files using Levenshtein distance.
- `calculate_internal_tree_entropy(filepath)` measures structural drift between equations of increasing complexity using SymPy and tree-edit distance.
- `printlatexequation(folder_path)` prints the LaTeX equation for a saved PySR run.

These helpers are not called by default. Example calls are left commented at the bottom of `Regression.py`.

## Generating Figures

Create the methodology visualization:

```powershell
python Opti_Figure.py
```

This saves:

```text
methodology_optimization_viz.png
```

Create the manually specified confusion matrix:

```powershell
python figure.py
```

Run the simple feasibility-region demonstration:

```powershell
python feasable.py
```

## Data and Output Notes

Several local datasets and generated output folders are ignored by Git:

- `.venv/`
- `AllSAMPLES.csv`
- `AllSAMPLES.xlsx`
- `my_equations/`
- `outputs/`
- `Result_Sigmoid_Function/`
- `R^2_Result/`
- `raw_scores.txt`

A fresh clone needs the dataset and any desired saved PySR runs copied into the project before previous experiments can be reproduced.

## Known Caveats

- The code uses the existing column spelling `Feasability`.
- `Regression.py` is configured by editing variables directly in `main()` rather than through command-line arguments.
- PySR training can take a long time, especially with `niterations=1000`.
- The train/validation/test split now uses a random seed each run, so exact results vary unless you replace `random.randint(...)` with a fixed seed.
- The optional MinMax scaling block is present but commented out.

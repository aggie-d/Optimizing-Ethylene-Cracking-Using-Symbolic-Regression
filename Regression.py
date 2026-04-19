import numpy as np
import pandas as pd
import os, time, random, sqlite3, re, json
from pysr import PySRRegressor
from math import sin, exp
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from Levenshtein import distance as lev_dist
from sklearn.preprocessing import MinMaxScaler
from zss import Node, simple_distance
from sympy import parse_expr, preorder_traversal
import matplotlib.pyplot as plt


def load_and_split_data():
    
    df = pd.read_excel('AllSAMPLES.xlsx')
    
    # random_int = random.randint(0, 2**9)
    random_int = 256
    # Rename columns here so it is consistent for all sets
    df.columns = ["Tin", "q", "flow_shale", "flow_steam", "length", "Pressure", "Status", "Feasability"]

    # 1. Split into Train (55%) and Temp (45%)
    df_train, df_temp = train_test_split(df, test_size=0.45, random_state=random_int, shuffle=True)

    # 2. Split Temp (45%) into Validation (15%) and Test (30%)
    # We use 0.67 because 67% of 45% = 30% of total
    df_val, df_test = train_test_split(df_temp, test_size=0.67, random_state=42, shuffle=True)

    return df_train, df_val, df_test, random_int

def Training_Set(df, var, run_id, path):

    # 1. Separate X (Features) and y (Target)
    # X contains passed columns *except* 'Feasability' and 'Status'
    X = df[var].to_numpy() 
    # y contains only the 'Feasability' column
    Y = df["Feasability"].to_numpy()

    model = PySRRegressor(
        maxsize=50,
        populations=50,
        niterations=1000,  #< Increase me for better results
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[
            "exp",       
            "square",
            "sqrt",
            "inv",
            "cube",
            "log",
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss='SigmoidLoss()',
        # ^ Custom loss function (julia syntax)
        output_directory= "my_equations",  # Where to save the equations
        run_id=run_id,  # Save equations to disk
        warm_start=True,  # Load from output_directory if possible
        model_selection="accuracy", # Select the most accurate model at the end
        nested_constraints={
            "square": {"square": 2, "sqrt": 4},
            "sqrt": {"square": 4, "sqrt": 2},
            "exp": {"exp": 0, "inv": 0},
            "cube": {"square": 1, "cube": 1, "exp": 0},
        },
        maxdepth=30,
        complexity_of_constants=2,
        early_stop_condition=(
        "stop_if(loss, complexity) = loss < 0.03 && complexity < 10"
        # Stop early if we find a good and simple equation
        ),
    )

	        # "log": {"exp": 1, "log": 0},
            # "sin": {"sin": 0, "cos": 0, "tan": 0}, 
            # "cos": {"sin": 0, "cos": 0, "tan": 0},
            # "tan": {"sin": 0, "cos": 0, "tan": 0},
    model.fit(X, Y, variable_names=var)
    


def Validation_Set(df, var, path):
    # 1. Separate X (Features) and y (Target)
    X_val = df[var].to_numpy() 
    y_val = df["Feasability"].to_numpy() 

    model = PySRRegressor.from_file(run_directory=path)
    model.set_params(extra_sympy_mappings="inv(x) = 1/x",)
    raw_scores_z = model.predict(X_val)
    
    # 2. Apply the same threshold as your Test Set
    predicted_classes = np.sign(raw_scores_z)
    predicted_classes[predicted_classes == 0] = -1 
    
    # 3. Calculate Accuracy
    acc = accuracy_score(y_val, predicted_classes)

    return acc
   


def Test_Set(df, var, matrix_status, path):

    X_test = df[var].to_numpy() 
    y_test = df["Feasability"].to_numpy()

    model = PySRRegressor.from_file(run_directory=path)
    model.set_params(extra_sympy_mappings={"inv": lambda x: 1 / x})
    lambda_func = model.get_best()
    latex_form = model.latex()

   
    # 1. Get the raw prediction scores (z values)
    # y_pred is the raw output (z) of the symbolic equation: f(X)
    raw_scores_z = model.predict(X_test)
    

    # 2. Apply a threshold (typically 0.5) to get binary predicted classes (0 or 1)
    # Use .astype(int) to convert True/False to 1/0
    # probabilities = 1 / (1 + np.exp(-raw_scores_z))
    clipped_z = np.clip(raw_scores_z, -500, 500)
    probabilities = 1 / (1 + np.exp(-clipped_z))

    predicted_classes = np.sign(raw_scores_z)
    predicted_classes[predicted_classes == 0] = -1  # Treat 0 as -1 for binary classification
    
    # --- INCORRECT PREDICTIONS ANALYSIS (Pure NumPy) ---
    # 4. Find the indices where the prediction doesn't match the actual label
    # incorrect_mask = predicted_classes != y_test
    # incorrect_indices = np.where(incorrect_mask)[0]

    # print(f"\n--- Incorrect Predictions Analysis ({len(incorrect_indices)} errors) ---")
    
    # if len(incorrect_indices) > 0:
    #     # Extract the data just for the incorrect rows
    #     inc_z = raw_scores_z[incorrect_mask]
    #     inc_prob = probabilities[incorrect_mask] * 100
    #     inc_pred = predicted_classes[incorrect_mask].astype(int)
    #     inc_actual = y_test[incorrect_mask].astype(int)

    #     # Sort by absolute raw score descending (most confident errors first)
    #     sort_order = np.argsort(-np.abs(inc_z))

    #     # Print the table header
    #     print(f"{'Index':<7} | {'Raw Score (z)':<15} | {'Probability':<15} | {'Predicted':<10} | {'Actual':<10}")
    #     print("-" * 68)
        
    #     # Print each incorrect row
    #     for i in sort_order:
    #         orig_idx = incorrect_indices[i]
    #         print(f"{orig_idx:<7} | {inc_z[i]:<15.4f} | {inc_prob[i]:<13.2f}% | {inc_pred[i]:<10} | {inc_actual[i]:<10}")
    # else:
    #     print("Wow, 100% accuracy! No errors to show.")
    
    # print("-" * 68 + "\n")


    # 4. Calculate the Accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes)
    recall = recall_score(y_test, predicted_classes)
    f1 = f1_score(y_test, predicted_classes)
    auc = roc_auc_score(y_test, probabilities)
    print(f"Precision Score: {precision:.4f} (Safety Focus - minimizing false positives)")
    print(f"Recall Score:    {recall:.4f} (Profit Focus - minimizing false negatives)")
    print(f"F1 Score:        {f1:.4f}")
    print(f"AUC:             {auc:.4f}")

    print(f"Accuracy Score: {accuracy:.4f}")
    class_names = ["Infeasible", "Feasible"]
    disp = ConfusionMatrixDisplay.from_predictions(
    y_test, 
    predicted_classes, 
    display_labels=class_names, 
    cmap= 'RdBu',      # This generates the exact blue color scheme from your image
    text_kw={'color': 'white'},
    colorbar=False     # Set to True if you want the color scale legend on the right
    )

    # Add a title if desired
    plt.title("PySR Model")

    # Display the plot
    plt.show()
    if matrix_status:
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_classes).ravel()
        print("-" * 30)
        print(f"Type 1 Errors (False Positives): {fp}")
        print(f"   (Model said Feasible, Reality was NOT)")
        print(f"Type 2 Errors (False Negatives): {fn}")
        print(f"   (Model said NOT Feasible, Reality was Feasible)")
        print("-" * 30)
        print(f"Correctly Identified Feasible:     {tp}")
        print(f"Correctly Identified Not Feasible: {tn}\n")

    return accuracy, lambda_func, latex_form

    
def print_results(accuracy, best_lambda_func, time_elapsed, var, rand_state, v_score, run_id, latex):
    try:
        connection = sqlite3.connect("Results.db")
        cursor = connection.cursor()

        complexity_val = int(best_lambda_func['complexity'])
        vars_serialized = json.dumps(var)

        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                trial_id TEXT PRIMARY KEY,
                accuracy REAL,
                validation REAL,
                variables_json TEXT,
                random_state INTEGER,
                time_elapsed REAL,
                complexity INTEGER,
                latex_format TEXT
            )
        ''')

        cursor.execute('''
            INSERT OR REPLACE INTO results 
            (trial_id, accuracy, validation, variables_json, random_state, time_elapsed, complexity, latex_format)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, accuracy, v_score, vars_serialized, rand_state, time_elapsed, complexity_val, latex))
        connection.commit()
        connection.close()
    except sqlite3.IntegrityError as e:
        print(f"Integrity error: {e}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")



    # folder_name = "Result_Sigmoid_Function"

    # var_list = []
    # for i in range(len(var)):
    #     var_list.append(f"X{i}: {var[i]}")

    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)  # Creates the folder if it doesn't exist

    # # Step 2: Create a text file inside the folder
    # result = f"\nAccuracy Score for the best PySR equation from {path}: {accuracy}\nValidation Score:{v_score}\nVariables Used: {var_list}\nRandom State Used: {rand_state}\nTime Elapsed: {time_elapsed:.2f} seconds\nEquation Used: {best_lambda_func}\n"
    # dashes = "-" * 90
    
    # txt = result + dashes
    # file_path = os.path.join(folder_name, "2_7_2026_Results.txt")
    # with open(file_path, "a") as file:
    #     file.write(txt + "\n")

def compare_pysr_runs(file1, file2, top_n=5):
    # 1. Load the Hall of Fame dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # 2. Get the top N equations (lowest loss)
    eqs1 = df1.nsmallest(top_n, 'Loss')['Equation'].tolist()
    eqs2 = df2.nsmallest(top_n, 'Loss')['Equation'].tolist()
    
    # 3. Calculate Cross-Model Distance Matrix
    dist_matrix = np.zeros((top_n, top_n))
    for i, e1 in enumerate(eqs1):
        for j, e2 in enumerate(eqs2):
            dist_matrix[i, j] = lev_dist(str(e1), str(e2))
            
    # 4. Calculate Metrics
    mean_dist = np.mean(dist_matrix)
    min_dist = np.min(dist_matrix) # Distance between the two "best" overall
    
    print(f"--- Comparison: {file1} vs {file2} ---")
    print(f"Mean Structural Divergence: {mean_dist:.2f}")
    print(f"Closest Match Distance: {min_dist:.2f}")
    
    return dist_matrix

def sympy_to_zss(expr):
    """Converts a SymPy expression into a ZSS Node tree."""
    if expr.is_Atom:
        return Node(str(expr))
    
    root = Node(str(expr.func))
    for arg in expr.args:
        root.addkid(sympy_to_zss(arg))
    return root

def calculate_internal_tree_entropy(filepath):
    df = pd.read_csv(filepath)
    # Ensure we use the correct column names
    eq_col = 'sympy_format' if 'sympy_format' in df.columns else 'Equation'
    
    # Sort by complexity to see the evolution
    df = df.sort_values('Complexity')
    equations = df[eq_col].tolist()
    complexities = df['Complexity'].tolist()
    
    print(f"--- Structural Evolution for {filepath} ---")
    
    distances = []
    for i in range(len(equations) - 1):
        tree1 = sympy_to_zss(parse_expr(equations[i]))
        tree2 = sympy_to_zss(parse_expr(equations[i+1]))
        
        dist = simple_distance(tree1, tree2)
        distances.append(dist)
        
        print(f"Comp {complexities[i]} -> {complexities[i+1]} | Distance: {dist}")

    avg_drift = sum(distances) / len(distances) if distances else 0
    print(f"\nAverage Structural Drift: {avg_drift:.2f}")
    # print(distances)

def start(variables, run_id, path, num_repeat):
    
    # for i in range(num_repeat):
    #     if path[-1] != ".":
    #        path = path[:-1] + str(i)
    #        run_id = run_id[:-1]+ str(i)
    #     else: 
    #         path = path + str(i)
    #         run_id = run_id + str(i)

        time_start = time.time()

        df_train, df_val, df_test, random_state_used = load_and_split_data()

        # 2. Initialize the MinMaxScaler
        # The range (1, 2) ensures all numbers are positive and strictly > 0
        # scaler = MinMaxScaler(feature_range=(1, 2))

        # 3. Fit on Train, then transform Train, Val, and Test
        # This overwrites the specific columns in place so your other functions work natively
        # df_train[variables] = scaler.fit_transform(df_train[variables])
        # df_val[variables] = scaler.transform(df_val[variables])
        # df_test[variables] = scaler.transform(df_test[variables])

        # Training_Set(df_train, variables, run_id, path)
        # validation_score= Validation_Set(df_val, variables, path)
        accuracy, best_lambda_func, latex = Test_Set(df_test, variables, True, path)

        time_end = time.time()
        time_elapsed = time_end - time_start
        # print_results(accuracy, best_lambda_func, time_elapsed, variables, random_state_used, validation_score, run_id, latex)

def printlatexequation(folder_path):
    model = PySRRegressor.from_file(run_directory=folder_path)
    print(f"{model.latex()}\n")


def main():
    path = "my_equations/2_16_26.0.5"
    run_id = "2_16_26.0.5"
    num_repeat = 1

    all_variables = ["Tin", "q", "flow_shale","flow_steam","length", "Pressure"]
    no_Tin = ["Q", "flow_shale","flow_steam","length", "Pressure"]
    
    modified = ["Q", "length",]

    start(all_variables, run_id, path, num_repeat)


    # path = "my_equations/2_9_26.1."
    # run_id = "2_9_26.1."
    # modified = ["Q", "flow_shale", "length", "Pressure"]
    
    # start(modified, run_id, path, num_repeat)

    # path = "my_equations/2_9_26.2."
    # run_id = "2_9_26.2."
    # modified = ["Q", "flow_steam", "length", "Pressure"]
    
    # start(modified, run_id, path, num_repeat)






    
    


if __name__ == "__main__":
    main()
    # compare_pysr_runs("my_equations/3_8_26.2.4/hall_of_fame.csv", "my_equations/3_8_26.2.2/hall_of_fame.csv")
    # calculate_internal_tree_entropy("my_equations/2_14_26.1.3/hall_of_fame.csv")
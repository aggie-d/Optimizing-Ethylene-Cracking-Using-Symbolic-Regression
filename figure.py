import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# 1. Define the data from the image
# Matrix: [True Label Row 0, True Label Row 1]
# Columns: Predicted Label (Infeasible, Feasible)
cm = np.array([[1831, 6], 
               [13, 1650]])

labels = ["Infeasible", "Feasible"]

# 2. Create the display object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# 3. Plot the matrix
fig, ax = plt.subplots(figsize=(6, 5))
# Use the Greens colormap to match the original
disp.plot(cmap="RdBu", ax=ax, colorbar=False)

# 4. Manually update the cell text to include the exact numbers and percentages
# Image layout: 
# Cell [0,0]: 1831 (99.3%) | Cell [0,1]: 6 (0.3%)
# Cell [1,0]: 13 (0.7%)   | Cell [1,1]: 1650 (99.6%)
custom_text = [
    ["695\n  96.2%", "27\n   3.8%"],
    ["15\n   1.9%", "771\n  98.1%"]
]

for i in range(len(labels)):
    for j in range(len(labels)):
        # Update the text for each cell
        text_element = disp.text_[i, j]
        text_element.set_text(custom_text[i][j])
        
        # Color adjustment for readability
        text_element.set_color("white")
        

# 5. Set title and labels
ax.set_title("PySR Model", pad=20, fontsize=14)
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)

plt.tight_layout()
plt.show()
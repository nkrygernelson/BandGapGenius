import matplotlib.pyplot as plt
import numpy as np
import ternary


# Set up the ternary plot
figure, tax = ternary.figure(scale=1.0)
tax.boundary(linewidth=2.0)
tax.gridlines(color="black", multiple=0.1)

# Define color map for the stability (formation energy)
cmap = plt.get_cmap("coolwarm")
norm = plt.Normalize(-0.2, 0.2)  # Adjust based on your data range

#time to make the real data. I have to create the chemical formula for all the possible combinations of the elements with a maximum for each element
# Plot data points
data = []


for (Sb, P, S, energy) in data:
    color = cmap(norm(energy))
    tax.scatter([(Sb, P, S)], color=color, label=f"{energy:.2f} eV/atom")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=tax.ax, orientation="vertical", label="Stability (eV/atom)")

# Labels and title
tax.left_axis_label("S", fontsize=12)
tax.right_axis_label("Ba", fontsize=12)
tax.bottom_axis_label("As", fontsize=12)
plt.title("Ternary Phase Diagram")

# Show plot
plt.tight_layout()
plt.show()

#!/usr/bin/env python
"""
viz_3d_synergy.py â€?3D Synergistic Response Surface

Nature Genetics 2026 Â· MetaRareEpi Framework

Displays epistatic synergy diverging violently from a flat linear
additive expectation plane. Fully complies with Nature's formatting:
- 'inferno' colormap for synergy surface
- 8pt Helvetica font
- No pane lines or grid
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def plot_3d_synergy_surface_nature(output_path: str):
    # Enforce Nature Portfolio typography
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(3.5, 3.0), dpi=600)  # ~89x76 mm (single column)
    ax = fig.add_subplot(111, projection='3d')
    
    # Define Mutation Burden Grid
    X, Y = np.meshgrid(np.linspace(0, 4, 50), np.linspace(0, 4, 50))
    
    # Mathematical representation: Linear additivity vs Synergistic spike
    Z_additive = 0.3 * X + 0.4 * Y
    Z_synergy = Z_additive + 2.8 * (X * Y)**2.5 * np.exp(-0.15 * (X+Y)) 
    
    # Plot transparent additive expectation plane
    ax.plot_surface(X, Y, Z_additive, color='#E0E0E0', alpha=0.5, label='Additive Base')
    
    # Plot solid synergistic surface
    ax.plot_surface(X, Y, Z_synergy, cmap='inferno', edgecolor='none', alpha=0.95)
    
    # Typography strictly to Nature's 8pt standard
    ax.set_xlabel('\nLDLR Constrained Burden', fontsize=8)
    ax.set_ylabel('\nPCSK9 Constrained Burden', fontsize=8)
    ax.set_zlabel('\nNormalized LDL-C Level', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Optimal aesthetic viewing angle for Nature
    ax.view_init(elev=28, azim=-55)
    
    # Clean up pane lines and grids for publication grade
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set pane colors to transparent
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    
    # Adjust layout to remove massive white margins
    plt.tight_layout(pad=1.0)
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), format='pdf', bbox_inches='tight', transparent=True)
    print(f"âœ?Nature 3D Synergy Surface saved to {output_path}")

if __name__ == "__main__":
    out = "viz/output/Figure_4B_3D_Surface_Nature.pdf"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    plot_3d_synergy_surface_nature(out)

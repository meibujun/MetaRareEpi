#!/usr/bin/env python
"""
viz_network.py — Figure 4: Epistatic Network Visualisation

Nature Genetics 2026 · MetaRareEpi Framework

Generates:
    Panel A: Epistatic hub network graph (human 1KGP)
    Panel B: Circos-style genomic interaction arc plot (bovine)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
from pathlib import Path


def generate_figure4(output_path: str = "viz/output/Figure_4_Network.pdf"):
    """Generate publication-quality epistatic network visualization."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['pdf.fonttype'] = 42

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), dpi=600)

    # ── Panel A: Network graph ──
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # Gene nodes
    genes = {
        'LDLR': (0.0, 1.0), 'PCSK9': (0.9, 0.3),
        'APOB': (0.6, -0.8), 'LIPA': (-0.6, -0.8),
        'NPC1L1': (-0.9, 0.3),
    }
    # Edges with interaction strength
    edges = [
        ('LDLR', 'PCSK9', 14), ('LDLR', 'APOB', 8),
        ('APOB', 'LIPA', 11), ('PCSK9', 'NPC1L1', 6),
        ('LDLR', 'NPC1L1', 5),
    ]

    # Draw edges
    for g1, g2, strength in edges:
        x1, y1 = genes[g1]
        x2, y2 = genes[g2]
        lw = strength / 5.0
        alpha = min(strength / 15.0, 0.9)
        ax.plot([x1, x2], [y1, y2], '-', color='#E64B35', linewidth=lw,
                alpha=alpha, zorder=1)

    # Draw nodes
    for gene, (x, y) in genes.items():
        circle = plt.Circle((x, y), 0.18, color='#4DBBD5', ec='#2D7D8E',
                           linewidth=1.0, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, gene, ha='center', va='center', fontsize=5.5,
                fontweight='bold', color='white', zorder=4)

    ax.set_title('A  Human Epistatic Hub (1KGP)', fontsize=8,
                 fontweight='bold', loc='left')
    ax.axis('off')

    # ── Panel B: Genomic interaction arcs (Circos-style) ──
    ax = axes[1]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # Chromosomes as arc segments
    n_chr = 29  # bovine autosomes
    angles = np.linspace(0, 2 * np.pi, n_chr + 1)[:-1]
    arc_colors = plt.cm.Set3(np.linspace(0, 1, n_chr))

    for i in range(n_chr):
        theta1 = np.degrees(angles[i])
        theta2 = np.degrees(angles[i]) + 360 / n_chr * 0.85
        arc = mpatches.Arc((0, 0), 2.2, 2.2, angle=0,
                          theta1=theta1, theta2=theta2,
                          linewidth=4, color=arc_colors[i])
        ax.add_patch(arc)
        # Label key chromosomes
        if i + 1 in [14, 19]:
            mid_angle = np.radians((theta1 + theta2) / 2)
            lx = 1.35 * np.cos(mid_angle)
            ly = 1.35 * np.sin(mid_angle)
            ax.text(lx, ly, f'BTA{i+1}', ha='center', va='center',
                    fontsize=5, fontweight='bold', color='#333')

    # Draw interaction arcs
    interactions = [
        (13, 18, '#E64B35', 2.5),  # BTA14-BTA19 (DGAT1-FASN)
        (4, 18, '#F39B7F', 1.0),
        (13, 25, '#F39B7F', 0.8),
    ]

    for chr1, chr2, color, width in interactions:
        a1 = (angles[chr1] + angles[chr1] + 2 * np.pi / n_chr * 0.85) / 2
        a2 = (angles[chr2] + angles[chr2] + 2 * np.pi / n_chr * 0.85) / 2
        x1, y1 = 0.95 * np.cos(a1), 0.95 * np.sin(a1)
        x2, y2 = 0.95 * np.cos(a2), 0.95 * np.sin(a2)
        # Bezier-like curve through center
        mid_x, mid_y = 0.3 * (x1 + x2), 0.3 * (y1 + y2)
        t = np.linspace(0, 1, 50)
        bx = (1-t)**2 * x1 + 2*(1-t)*t * mid_x + t**2 * x2
        by = (1-t)**2 * y1 + 2*(1-t)*t * mid_y + t**2 * y2
        ax.plot(bx, by, '-', color=color, linewidth=width, alpha=0.8)

    ax.set_title('B  Bovine Epistatic Network', fontsize=8,
                 fontweight='bold', loc='left')
    ax.axis('off')

    plt.tight_layout(pad=1.0)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), format='pdf', bbox_inches='tight', transparent=True)
    print(f"✓ Figure 4 saved to {output_path}")


if __name__ == "__main__":
    out = "viz/output/Figure_4_Network.pdf"
    if len(sys.argv) > 1:
        out = sys.argv[1]
    generate_figure4(out)

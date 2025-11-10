"""
F1 Dynamic Systems Model Diagram Generator
Generates Figure 3 for academic paper - comprehensive visual model of the system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_system_diagram():
    """
    Creates a professional system diagram showing:
    - State variables
    - Control inputs
    - State transitions
    - Output equation
    - Objective function
    """

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'F1 Race Strategy Dynamic Systems Model',
            fontsize=18, fontweight='bold', ha='center', va='top')
    ax.text(5, 9.1, 'Discrete-Time State-Space Representation',
            fontsize=12, ha='center', va='top', style='italic')

    # ============ STATE VARIABLES BOX ============
    state_box = FancyBboxPatch((0.5, 6.5), 2.5, 2,
                               boxstyle="round,pad=0.1",
                               edgecolor='#2E86AB', facecolor='#A9D6E5',
                               linewidth=2.5)
    ax.add_patch(state_box)

    ax.text(1.75, 8.2, 'State Variables', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.75, 7.95, r'$\mathbf{x}(t)$', fontsize=14, ha='center', style='italic')
    ax.text(1.75, 7.6, r'$x_1$: Tire age (laps)', fontsize=9, ha='center', family='monospace')
    ax.text(1.75, 7.35, r'$x_2$: Track wetness [0,1]', fontsize=9, ha='center', family='monospace')
    ax.text(1.75, 7.1, r'$x_3$: Cumulative wear', fontsize=9, ha='center', family='monospace')
    ax.text(1.75, 6.85, r'$x_4$: Fuel load (kg)', fontsize=9, ha='center', family='monospace')

    # ============ CONTROL INPUTS BOX ============
    control_box = FancyBboxPatch((0.5, 4.2), 2.5, 1.8,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#6A994E', facecolor='#C7EFCF',
                                 linewidth=2.5)
    ax.add_patch(control_box)

    ax.text(1.75, 5.75, 'Control Inputs', fontsize=11, fontweight='bold', ha='center')
    ax.text(1.75, 5.5, r'$\mathbf{u}(t)$', fontsize=14, ha='center', style='italic')
    ax.text(1.75, 5.15, r'$u_1$: Pit decision {0,1}', fontsize=9, ha='center', family='monospace')
    ax.text(1.75, 4.9, r'$u_2$: Compound choice', fontsize=9, ha='center', family='monospace')
    ax.text(1.75, 4.65, '     {SOFT, MED, HARD, INT}', fontsize=8, ha='center', family='monospace')

    # ============ STATE TRANSITION BOX (CENTER) ============
    transition_box = FancyBboxPatch((3.5, 5.0), 3.0, 3.0,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='#BC4749', facecolor='#F2C6A7',
                                    linewidth=2.5)
    ax.add_patch(transition_box)

    ax.text(5, 7.75, 'State Transitions', fontsize=11, fontweight='bold', ha='center')
    ax.text(5, 7.45, r'$\mathbf{x}(t+1) = f[\mathbf{x}(t), \mathbf{u}(t)]$',
            fontsize=11, ha='center', style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Individual transition equations (simplified for matplotlib)
    ax.text(5, 7.0, r'Tire age: $x_1(t+1) = x_1(t) + 1$ if no pit, else $0$',
            fontsize=8.5, ha='center')

    ax.text(5, 6.5, r'Wetness: $x_2(t+1) = x_2(t) \cdot e^{-\lambda_{dry}}$',
            fontsize=8.5, ha='center')

    ax.text(5, 6.0, r'Wear: $x_3(t+1) = x_3(t) + \delta(u_2) \cdot x_1(t)$',
            fontsize=8.5, ha='center')

    ax.text(5, 5.5, r'Fuel: $x_4(t+1) = x_4(t) - \gamma_{fuel}$',
            fontsize=8.5, ha='center')

    # ============ OUTPUT BOX ============
    output_box = FancyBboxPatch((7.0, 6.0), 2.5, 2.0,
                                boxstyle="round,pad=0.1",
                                edgecolor='#8E44AD', facecolor='#D4B3E8',
                                linewidth=2.5)
    ax.add_patch(output_box)

    ax.text(8.25, 7.75, 'Performance Output', fontsize=11, fontweight='bold', ha='center')
    ax.text(8.25, 7.5, r'$y(t)$: Lap Time', fontsize=10, ha='center', style='italic')
    ax.text(8.25, 7.15, r'$y(t) = t_{base} \cdot \phi[\mathbf{x}]$', fontsize=9, ha='center', family='monospace')
    ax.text(8.25, 6.75, r'$+ \beta_{fuel} \cdot x_4(t)$', fontsize=9, ha='center', family='monospace')
    ax.text(8.25, 6.35, r'$+ P_{pit} \cdot u_1(t)$', fontsize=9, ha='center', family='monospace')

    # ============ OBJECTIVE FUNCTION BOX ============
    objective_box = FancyBboxPatch((3.5, 1.5), 3.0, 1.3,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='#E63946', facecolor='#FFD6D9',
                                   linewidth=2.5)
    ax.add_patch(objective_box)

    ax.text(5, 2.55, 'Optimization Objective', fontsize=11, fontweight='bold', ha='center')
    ax.text(5, 2.2, r'$\min_{\mathbf{u}} \quad J = \sum_{t=1}^{T} y(t)$',
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.text(5, 1.85, 'Minimize total race time over T laps', fontsize=8, ha='center', style='italic')

    # ============ ARROWS ============
    # State -> Transition
    arrow1 = FancyArrowPatch((3.0, 7.5), (3.5, 7.0),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='#2E86AB')
    ax.add_patch(arrow1)

    # Control -> Transition
    arrow2 = FancyArrowPatch((3.0, 5.5), (3.5, 6.0),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='#6A994E')
    ax.add_patch(arrow2)

    # Transition -> Output
    arrow3 = FancyArrowPatch((6.5, 7.0), (7.0, 7.0),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='#BC4749')
    ax.add_patch(arrow3)

    # Output -> Objective (curved)
    arrow4 = FancyArrowPatch((8.25, 6.0), (6.5, 2.8),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2.5, color='#8E44AD',
                            connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow4)

    # Feedback loop: Transition -> State (next timestep)
    arrow5 = FancyArrowPatch((5.0, 5.0), (1.75, 6.5),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=2, color='#BC4749', linestyle='--',
                            connectionstyle="arc3,rad=-0.5")
    ax.add_patch(arrow5)
    ax.text(2.5, 5.5, r'$x(t+1)$', fontsize=10, style='italic', color='#BC4749')

    # ============ PARAMETER ANNOTATIONS ============
    param_box = FancyBboxPatch((0.3, 0.2), 9.4, 1.0,
                               boxstyle="round,pad=0.05",
                               edgecolor='gray', facecolor='#F8F9FA',
                               linewidth=1, linestyle='--')
    ax.add_patch(param_box)

    ax.text(5, 1.0, 'Model Parameters (Calibrated on Monaco 2023)',
            fontsize=9, ha='center', fontweight='bold')
    ax.text(5, 0.7, r'$\lambda_{dry}=0.070$ (drying rate) | ' +
                    r'$\delta_{SOFT}=0.0007$ (tire deg.) | ' +
                    r'$\beta_{fuel}=0.05$ s/kg | ' +
                    r'$P_{pit}=22.0$ s | ' +
                    r'$t_{base}=75.0$ s',
            fontsize=8, ha='center', family='monospace')

    # ============ LEGEND ============
    ax.text(0.5, 3.7, 'System Type:', fontsize=8, fontweight='bold')
    ax.text(0.5, 3.45, '• Discrete-time', fontsize=7)
    ax.text(0.5, 3.25, '• Nonlinear', fontsize=7)
    ax.text(0.5, 3.05, '• Time-varying', fontsize=7)
    ax.text(0.5, 2.85, '• Constrained', fontsize=7)

    ax.text(7.5, 3.7, 'Applications:', fontsize=8, fontweight='bold')
    ax.text(7.5, 3.45, '• Strategy optimization', fontsize=7)
    ax.text(7.5, 3.25, '• Scenario analysis', fontsize=7)
    ax.text(7.5, 3.05, '• Real-time decision', fontsize=7)

    plt.tight_layout()

    # Save figure
    output_path = 'presentation_graphs/3_system_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] System diagram saved: {output_path}")

    # Also save high-res version for paper
    output_path_hires = 'presentation_graphs/3_system_diagram_hires.png'
    plt.savefig(output_path_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"[OK] High-res diagram saved: {output_path_hires}")

    plt.close()

if __name__ == "__main__":
    print("================================================================================")
    print("F1 DYNAMIC SYSTEMS MODEL - DIAGRAM GENERATOR")
    print("================================================================================")
    print("\nGenerating Figure 3: System Diagram for Academic Paper")
    print("This diagram illustrates the discrete-time state-space model used for")
    print("race strategy optimization.\n")

    create_system_diagram()

    print("\n================================================================================")
    print("DIAGRAM GENERATION COMPLETE")
    print("================================================================================")
    print("\nFigure 3 shows:")
    print("  • State Variables: tire_age, wetness, wear, fuel")
    print("  • Control Inputs: pit_decision, compound_choice")
    print("  • State Transitions: x(t+1) = f[x(t), u(t)]")
    print("  • Output Equation: lap_time(t)")
    print("  • Optimization Objective: minimize total race time")
    print("\nReady for inclusion in academic paper.")

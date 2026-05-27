import matplotlib.pyplot as plt
import numpy as np

TACTIQ_BG = '#111'
TACTIQ_FG = '#eee'

def test_donut():
    fig = plt.figure(figsize=(4, 5))
    fig.patch.set_facecolor(TACTIQ_BG)
    ax_side = fig.add_subplot(111)
    ax_side.set_facecolor(TACTIQ_BG)
    ax_side.axis('off')
    
    # Data
    p_shot = 15.4
    p_retained = 61.5
    p_lost = 23.1
    total_gains = 65
    
    percentages = [p_shot, p_retained, p_lost]
    colors = ['#22c55e', '#3b82f6', '#f97316']
    
    # Filter 0% values
    pie_pcts = []
    pie_colors = []
    pie_labels = []
    
    label_map = {0: "Shot", 1: "Retained", 2: "Lost"}
    for idx, p in enumerate(percentages):
        if p > 0:
            pie_pcts.append(p)
            pie_colors.append(colors[idx])
            pie_labels.append(f"{label_map[idx]}\n{p}%")
            
    # Draw pie
    wedges, texts = ax_side.pie(
        pie_pcts,
        labels=pie_labels,
        colors=pie_colors,
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor=TACTIQ_BG, linewidth=3),
        textprops=dict(color='white', fontsize=9, fontweight='bold'),
        center=(0, 0.1)
    )
    
    ax_side.set_aspect('equal')
    ax_side.text(0, 0.1, f"{total_gains}\ngains", color='white', fontsize=13, fontweight='bold', ha='center', va='center')
    ax_side.text(0, 1.1, "Transition Outcomes (10s)", color='white', fontsize=12, fontweight='bold', ha='center', va='center')
    ax_side.text(0, -0.9, f"Total Transitions: {total_gains}", color='#aaa', fontsize=10, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig("scratch/test_donut.png")
    print("Donut tested and saved!")

if __name__ == "__main__":
    test_donut()

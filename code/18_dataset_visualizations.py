"""
Dataset Visualizations for PAN16 Author Profiling and Winogender datasets.
Generates 4 figures:
  1. PAN16 Age x Gender distribution (grouped bar chart)
  2. PAN16 Task label distribution by gender (grouped bar chart)
  3. PAN16 Train / Validation / Test split sizes (horizontal bar chart)
  4. Winogender contrastive pair structure schematic
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from utils.paths import PROJECT_ROOT, RAW_DATA_DIR

# ---------------------------------------------------------------------------
# Configuration - matches file 17 font sizes
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

AGE_LABELS = {0: '18–24', 1: '25–34', 2: '35–49', 3: '50–64', 4: '65+',
              '18-24': '18–24', '25-34': '25–34', '35-49': '35–49', '50-64': '50–64', '65-xx': '65+'}
AGE_ORDER = ['18-24', '25-34', '35-49', '50-64', '65-xx',  # string keys from pickle
             0, 1, 2, 3, 4]                                  # numeric keys from TSV
GENDER_COLORS = {'male': '#2E86AB', 'female': '#A23B72'}

OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "dataset_visualizations"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pan16():
    """Load all PAN16 splits from pickle files (clean, canonical source)."""
    pickle_dir = RAW_DATA_DIR / "pan16_raw" / "pickle_format"
    train = pd.DataFrame(pd.read_pickle(pickle_dir / "train.pkl"))
    val = pd.DataFrame(pd.read_pickle(pickle_dir / "validation.pkl"))
    test = pd.DataFrame(pd.read_pickle(pickle_dir / "test.pkl"))
    return train, val, test


def load_winogender():
    """Load Winogender contrastive pairs."""
    return pd.read_csv(RAW_DATA_DIR / "winogender_raw" / "counterfactual_winogender.csv")


# ---------------------------------------------------------------------------
# Plot 1 - Age × Gender Distribution
# ---------------------------------------------------------------------------

def plot_age_gender_distribution(train, val, test, output_dir):
    """
    Grouped bar chart: sample count per age group, split by gender.
    Uses the full dataset (train + val + test) to show overall composition.
    """
    df = pd.concat([train, val, test], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    age_groups = [a for a in AGE_ORDER if a in df['age'].unique()]
    x_positions = range(len(age_groups))
    bar_width = 0.35

    male_counts = [len(df[(df['age'] == a) & (df['gender'] == 'male')]) for a in age_groups]
    female_counts = [len(df[(df['age'] == a) & (df['gender'] == 'female')]) for a in age_groups]

    bars_m = ax.bar([x - bar_width / 2 for x in x_positions], male_counts,
                    bar_width, label='Male', color=GENDER_COLORS['male'],
                    edgecolor='black', linewidth=0.8)
    bars_f = ax.bar([x + bar_width / 2 for x in x_positions], female_counts,
                    bar_width, label='Female', color=GENDER_COLORS['female'],
                    edgecolor='black', linewidth=0.8)

    # Value labels
    for bar in list(bars_m) + list(bars_f):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 100,
                f'{int(h):,}', ha='center', va='bottom', fontsize=13, weight='bold')

    ax.set_xlabel('Age Group', fontsize=16, weight='bold')
    ax.set_ylabel('Number of Tweets', fontsize=16, weight='bold')
    ax.set_title('PAN16: Sample Distribution by Age Group and Gender',
                 fontsize=18, weight='bold', pad=15)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([AGE_LABELS[a] for a in age_groups], fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'pan16_age_gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 1 - Age and Gender distribution")


# ---------------------------------------------------------------------------
# Plot 2 - Task Label Distribution by Gender
# ---------------------------------------------------------------------------

def plot_task_by_gender(train, val, test, output_dir):
    """
    Grouped bar chart showing mention (task=1) vs. no-mention (task=0)
    counts for each gender, revealing potential spurious correlations.
    """
    df = pd.concat([train, val, test], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 10))

    genders = ['male', 'female']
    bar_width = 0.3
    x_positions = range(len(genders))

    no_mention = [len(df[(df['gender'] == g) & (df['task_label'] == 0)]) for g in genders]
    mention = [len(df[(df['gender'] == g) & (df['task_label'] == 1)]) for g in genders]

    bars_0 = ax.bar([x - bar_width / 2 for x in x_positions], no_mention,
                    bar_width, label='No Mention (0)', color='#5DADE2',
                    edgecolor='black', linewidth=0.8)
    bars_1 = ax.bar([x + bar_width / 2 for x in x_positions], mention,
                    bar_width, label='Contains Mention (1)', color='#F39C12',
                    edgecolor='black', linewidth=0.8)

    # Value labels with percentages
    for bars, counts in [(bars_0, no_mention), (bars_1, mention)]:
        for bar, count, g in zip(bars, counts, genders):
            total = len(df[df['gender'] == g])
            pct = count / total * 100
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 100,
                    f'{int(count):,}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontsize=13, weight='bold')

    ax.set_xlabel('Author Gender', fontsize=16, weight='bold')
    ax.set_ylabel('Number of Tweets', fontsize=16, weight='bold')
    ax.set_title('PAN16: Task Label Distribution by Gender\n'
                 '(Differences indicate potential spurious correlations)',
                 fontsize=18, weight='bold', pad=15)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(['Male', 'Female'], fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'pan16_task_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 2 - Task label by gender")


# ---------------------------------------------------------------------------
# Plot 3 - Train / Validation / Test Split Sizes
# ---------------------------------------------------------------------------

def plot_split_sizes(train, val, test, output_dir):
    """
    Horizontal bar chart showing the number of samples in each split.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    splits = ['Test', 'Validation', 'Train']
    counts = [len(test), len(val), len(train)]
    authors = [len(test) // 100, len(val) // 100, len(train) // 100]
    colors = ['#E74C3C', '#F39C12', '#2ECC71']

    bars = ax.barh(splits, counts, color=colors, edgecolor='black', linewidth=0.8, height=0.55)

    for bar, count, n_auth in zip(bars, counts, authors):
        ax.text(bar.get_width() + 300, bar.get_y() + bar.get_height() / 2,
                f'{count:,} tweets  ({n_auth} authors)',
                ha='left', va='center', fontsize=14, weight='bold')

    ax.set_xlabel('Number of Tweets', fontsize=16, weight='bold')
    ax.set_title('PAN16: Dataset Split Sizes\n(100 tweets per author)',
                 fontsize=18, weight='bold', pad=15)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=13)
    ax.set_xlim(0, max(counts) * 1.35)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'pan16_split_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 3 - Split sizes")


# ---------------------------------------------------------------------------
# Plot 4 - Winogender Contrastive Pair Structure
# ---------------------------------------------------------------------------

def plot_winogender_structure(wg, output_dir):
    """
    Schematic figure illustrating how Winogender contrastive pairs work.
    Shows sentence templates with the pronoun slot highlighted.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Winogender: Contrastive Pair Structure',
            ha='center', va='top', fontsize=20, weight='bold')
    ax.text(5, 9.0, f'120 sentence pairs  ·  pronoun swap only  ·  identical syntax',
            ha='center', va='top', fontsize=14, color='#555555')

    # --- Example 1 ---
    y1 = 7.5
    ax.text(0.5, y1, 'Example 1', fontsize=16, weight='bold', color='#2E86AB')
    ax.text(0.5, y1 - 0.55, 'Original:', fontsize=13, weight='bold', color='#333333')
    ax.text(2.3, y1 - 0.55,
            'The technician told the customer that ', fontsize=13, color='#333333')
    ax.text(7.55, y1 - 0.55, 'he', fontsize=14, weight='bold', color='#2E86AB',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#D6EAF8', edgecolor='#2E86AB', linewidth=1.5))
    ax.text(7.95, y1 - 0.55, ' could pay with cash.', fontsize=13, color='#333333')

    ax.text(0.5, y1 - 1.15, 'Counterfactual:', fontsize=13, weight='bold', color='#333333')
    ax.text(2.8, y1 - 1.15,
            'The technician told the customer that ', fontsize=13, color='#333333')
    ax.text(8.05, y1 - 1.15, 'she', fontsize=14, weight='bold', color='#A23B72',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5D5E7', edgecolor='#A23B72', linewidth=1.5))
    ax.text(8.6, y1 - 1.15, ' could pay with cash.', fontsize=13, color='#333333')

    # --- Example 2 ---
    y2 = 5.2
    ax.text(0.5, y2, 'Example 2', fontsize=16, weight='bold', color='#2E86AB')
    ax.text(0.5, y2 - 0.55, 'Original:', fontsize=13, weight='bold', color='#333333')
    ax.text(2.3, y2 - 0.55,
            'The engineer informed the client that ', fontsize=13, color='#333333')
    ax.text(7.35, y2 - 0.55, 'she', fontsize=14, weight='bold', color='#A23B72',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F5D5E7', edgecolor='#A23B72', linewidth=1.5))
    ax.text(7.9, y2 - 0.55, ' would need more time.', fontsize=13, color='#333333')

    ax.text(0.5, y2 - 1.15, 'Counterfactual:', fontsize=13, weight='bold', color='#333333')
    ax.text(2.8, y2 - 1.15,
            'The engineer informed the client that ', fontsize=13, color='#333333')
    ax.text(7.85, y2 - 1.15, 'he', fontsize=14, weight='bold', color='#2E86AB',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#D6EAF8', edgecolor='#2E86AB', linewidth=1.5))
    ax.text(8.25, y2 - 1.15, ' would need more time.', fontsize=13, color='#333333')

    # --- Key insight box ---
    box_y = 2.5
    box = mpatches.FancyBboxPatch((1.0, box_y - 0.8), 8, 1.8,
                                   boxstyle='round,pad=0.3',
                                   facecolor='#FDEBD0', edgecolor='#E67E22',
                                   linewidth=2)
    ax.add_patch(box)
    ax.text(5, box_y + 0.6, 'Key Property', ha='center', fontsize=16, weight='bold', color='#E67E22')
    ax.text(5, box_y + 0.0,
            'Every pair is syntactically identical - only the gendered pronoun differs.',
            ha='center', fontsize=14, color='#333333')
    ax.text(5, box_y - 0.45,
            'This isolates gender-specific activation patterns free of lexical confounds.',
            ha='center', fontsize=14, color='#333333')

    # Legend patches
    male_patch = mpatches.Patch(facecolor='#D6EAF8', edgecolor='#2E86AB',
                                linewidth=1.5, label='Male pronoun (he/him/his)')
    female_patch = mpatches.Patch(facecolor='#F5D5E7', edgecolor='#A23B72',
                                  linewidth=1.5, label='Female pronoun (she/her/hers)')
    ax.legend(handles=[male_patch, female_patch], loc='lower center',
              fontsize=13, frameon=True, fancybox=True, shadow=True, ncol=2,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(output_dir / 'winogender_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 4 - Winogender structure schematic")


# ---------------------------------------------------------------------------
# Plot 5 — Three-Head Model Architecture Diagram
# ---------------------------------------------------------------------------

def _draw_box(ax, cx, cy, w, h, text, facecolor, edgecolor, fontsize=13,
              text_color='black', bold=False, linewidth=2, second_line=None):
    """Helper: draw a rounded rectangle with centred text."""
    box = mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                   boxstyle='round,pad=0.15',
                                   facecolor=facecolor, edgecolor=edgecolor,
                                   linewidth=linewidth)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    if second_line:
        ax.text(cx, cy + 0.18, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, weight=weight)
        ax.text(cx, cy - 0.22, second_line, ha='center', va='center',
                fontsize=fontsize - 2, color='#555555', style='italic')
    else:
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, weight=weight)
    return cx, cy


def _draw_arrow(ax, x1, y1, x2, y2, color='#333333', lw=2):
    """Helper: draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


def plot_model_architecture(output_dir):
    """
    Clean architecture diagram for the three-head model:
    Input → Frozen Encoder → CLS (768) → Shared Layer (256, ReLU) → 3 Heads
    """
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.6, 'Three-Head Model Architecture',
            ha='center', va='top', fontsize=22, weight='bold')
    ax.text(7, 9.1, 'Frozen encoder  +  Shared bottleneck  +  Three classification heads',
            ha='center', va='top', fontsize=14, color='#555555')

    # ── Layer 1: Input ──
    _draw_box(ax, 7, 8.0, 4.0, 0.8, 'Input Tweet', '#EBF5FB', '#2E86AB',
              fontsize=15, bold=True)

    _draw_arrow(ax, 7, 7.6, 7, 7.15)

    # ── Layer 2: Frozen Encoder ──
    # Outer "frozen" wrapper
    frozen_box = mpatches.FancyBboxPatch((3.5, 6.0), 7.0, 1.1,
                                          boxstyle='round,pad=0.2',
                                          facecolor='#F8F9FA', edgecolor='#AAB7B8',
                                          linewidth=2.5, linestyle='--')
    ax.add_patch(frozen_box)
    ax.text(10, 6.95, '(frozen)', fontsize=11, color='#AAB7B8', style='italic')
    _draw_box(ax, 7, 6.55, 5.5, 0.7, 'BERT / ModernBERT Encoder', '#D5F5E3', '#27AE60',
              fontsize=15, bold=True)

    _draw_arrow(ax, 7, 5.95, 7, 5.45)

    # ── Layer 3: CLS Token ──
    _draw_box(ax, 7, 5.1, 3.5, 0.6, 'CLS Token Representation', '#FDEBD0', '#E67E22',
              fontsize=14, bold=True, second_line='768 dimensions')

    _draw_arrow(ax, 7, 4.6, 7, 4.1)

    # ── Layer 4: Shared Intermediate Layer ──
    _draw_box(ax, 7, 3.7, 4.0, 0.7, 'Shared Intermediate Layer', '#F5EEF8', '#8E44AD',
              fontsize=14, bold=True, second_line='768 → 256, ReLU')

    # ── Arrows from shared layer to three heads ──
    head_y = 2.15
    _draw_arrow(ax, 5.8, 3.3, 3.0, head_y + 0.45)   # left
    _draw_arrow(ax, 7.0, 3.3, 7.0, head_y + 0.45)   # center
    _draw_arrow(ax, 8.2, 3.3, 11.0, head_y + 0.45)  # right

    # ── Layer 5: Three Classification Heads ──
    # Task head (center)
    _draw_box(ax, 7.0, head_y, 3.0, 0.8,
              'Task Head', '#D6EAF8', '#2E86AB',
              fontsize=14, bold=True, second_line='256 → 2 (mention)')

    # Gender head (left)
    _draw_box(ax, 3.0, head_y, 3.0, 0.8,
              'Gender Probe Head', '#F5D5E7', '#A23B72',
              fontsize=14, bold=True, second_line='256 → 2 (male / female)')

    # Age head (right)
    _draw_box(ax, 11.0, head_y, 3.0, 0.8,
              'Age Probe Head', '#FCF3CF', '#D4AC0D',
              fontsize=14, bold=True, second_line='256 → 5 (age groups)')

    # ── Loss weights annotation ──
    ax.text(7.0, 1.1, 'Joint Loss  =  1.0 × Task Loss  +  0.5 × Gender Loss  +  0.5 × Age Loss',
            ha='center', va='center', fontsize=13, color='#333333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA',
                      edgecolor='#CCCCCC', linewidth=1.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'three_head_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] Plot 5 - Three-head model architecture")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("Dataset Visualizations")
    print("=" * 70 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[1/2] Loading datasets...")
    train, val, test = load_pan16()
    wg = load_winogender()
    print(f"  PAN16:  {len(train)+len(val)+len(test):,} total tweets "
          f"({len(train):,} train / {len(val):,} val / {len(test):,} test)")
    print(f"  Winogender: {len(wg)} contrastive pairs\n")

    # Generate plots
    print("[2/2] Generating plots...")
    plot_age_gender_distribution(train, val, test, OUTPUT_DIR)
    plot_task_by_gender(train, val, test, OUTPUT_DIR)
    plot_split_sizes(train, val, test, OUTPUT_DIR)
    plot_winogender_structure(wg, OUTPUT_DIR)
    plot_model_architecture(OUTPUT_DIR)

    print(f"\n[OK] All dataset visualizations saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

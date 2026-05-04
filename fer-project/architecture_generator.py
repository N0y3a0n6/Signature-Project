"""Generate FER project architecture infographic."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    'bg':        '#0f1117',
    'panel':     '#1a1d27',
    'border':    '#2d3148',
    'data':      '#1e3a5f',
    'data_b':    '#3a7bd5',
    'model':     '#1e4d2b',
    'model_b':   '#38a169',
    'train':     '#4a2060',
    'train_b':   '#9b59b6',
    'eval':      '#4a3000',
    'eval_b':    '#f39c12',
    'demo':      '#1e3a4a',
    'demo_b':    '#00bcd4',
    'arrow':     '#5a6070',
    'arrow_hi':  '#7f8fa6',
    'text':      '#e8eaf0',
    'subtext':   '#a0a8c0',
    'accent':    '#ffffff',
    'emotion':   ['#e74c3c','#8e44ad','#2980b9','#f1c40f','#3498db','#e67e22','#95a5a6'],
}

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']

fig = plt.figure(figsize=(24, 15), facecolor=C['bg'])
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 24)
ax.set_ylim(0, 15)
ax.axis('off')
ax.set_facecolor(C['bg'])

# ── Helpers ────────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, fc, ec, radius=0.25, alpha=0.92, lw=1.5, zorder=3):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(b)
    return b

def txt(ax, x, y, s, size=9, color=C['text'], weight='normal',
        ha='center', va='center', zorder=5, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=zorder, **kw)

def arrow(ax, x0, y0, x1, y1, color=C['arrow'], lw=1.5,
          arrowstyle='->', mutation_scale=14, zorder=2, **kw):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, mutation_scale=mutation_scale), zorder=zorder, **kw)

def section_header(ax, x, y, w, label, color_b, color_f):
    box(ax, x, y, w, 0.55, fc=color_f, ec=color_b, radius=0.2, lw=2)
    txt(ax, x + w/2, y + 0.275, label, size=10, weight='bold',
        color=C['accent'])

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
txt(ax, 12, 14.4, 'Advanced Facial Expression Recognition — System Architecture',
    size=16, weight='bold', color=C['accent'])
txt(ax, 12, 13.9,
    'EfficientNet-B2  ·  FER2013 + CK+48 + JAFFE  ·  7 Emotion Classes  ·  Apple M1 MPS',
    size=9.5, color=C['subtext'])

# thin separator
ax.axhline(13.65, color=C['border'], lw=1.0, zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 1 — DATA PIPELINE  (x: 0.3 → 5.5)
# ══════════════════════════════════════════════════════════════════════════════
COL1_X  = 0.35
COL1_W  = 5.2
COL1_CX = COL1_X + COL1_W / 2

section_header(ax, COL1_X, 13.0, COL1_W, '① DATA PIPELINE', C['data_b'], C['data'])

# --- Raw datasets
box(ax, COL1_X, 11.6, COL1_W, 1.25, C['data'], C['data_b'], radius=0.2)
txt(ax, COL1_CX, 12.5, 'Raw Datasets', size=9.5, weight='bold', color=C['data_b'])
ds = [('FER2013','28,709 imgs','in-the-wild'),
      ('CK+48','981 imgs','lab posed'),
      ('JAFFE','213 imgs','Japanese female')]
for i,(name,count,note) in enumerate(ds):
    xi = COL1_X + 0.35 + i*1.65
    box(ax, xi, 11.7, 1.5, 0.7, '#0a2040', C['data_b'], radius=0.15, lw=1)
    txt(ax, xi+0.75, 12.22, name, size=8.5, weight='bold', color=C['data_b'])
    txt(ax, xi+0.75, 11.97, count, size=7.5, color=C['subtext'])
    txt(ax, xi+0.75, 11.77, note, size=6.5, color=C['subtext'],
        style='italic')

arrow(ax, COL1_CX, 11.6, COL1_CX, 11.05)

# --- Data threshing
box(ax, COL1_X, 9.55, COL1_W, 1.4, C['data'], C['data_b'], radius=0.2)
txt(ax, COL1_CX, 10.6, 'Data Threshing  (02_data_threshing.ipynb)', size=9, weight='bold', color=C['data_b'])
steps = ['Face detection (Haar)', 'Quality filter (brightness/blur/contrast)', 'Class balancing (undersample)']
for i, s in enumerate(steps):
    txt(ax, COL1_X+0.4, 10.28-i*0.24, f'▸ {s}', size=7.8, color=C['subtext'], ha='left')

arrow(ax, COL1_CX, 9.55, COL1_CX, 9.0)

# --- WeightedRandomSampler
box(ax, COL1_X, 8.15, COL1_W, 0.72, C['data'], C['data_b'], radius=0.2)
txt(ax, COL1_CX, 8.59, 'WeightedRandomSampler', size=9, weight='bold', color=C['data_b'])
txt(ax, COL1_CX, 8.3, 'Oversample rare classes (disgust, fear) per batch', size=7.8, color=C['subtext'])

arrow(ax, COL1_CX, 8.15, COL1_CX, 7.6)

# --- DataLoader
box(ax, COL1_X, 6.75, COL1_W, 0.72, C['data'], C['data_b'], radius=0.2)
txt(ax, COL1_CX, 7.19, 'DataLoader', size=9, weight='bold', color=C['data_b'])
txt(ax, COL1_CX, 6.9, 'batch=16  ·  num_workers=0  ·  224×224 RGB', size=7.8, color=C['subtext'])

# Augmentation callout
box(ax, COL1_X, 5.65, COL1_W, 0.95, '#0a1a30', C['data_b'], radius=0.15, lw=1, alpha=0.7)
txt(ax, COL1_CX, 6.35, 'Train Augmentation', size=8.5, weight='bold', color=C['data_b'])
txt(ax, COL1_CX, 6.1,  'H-flip p=0.5  ·  Rotate ±15°', size=7.5, color=C['subtext'])
txt(ax, COL1_CX, 5.85, 'Brightness [0.8,1.2]  ·  Zoom 0.1  ·  Shift 0.1', size=7.5, color=C['subtext'])

# Norm callout
box(ax, COL1_X, 4.85, COL1_W, 0.65, '#0a1a30', C['data_b'], radius=0.15, lw=1, alpha=0.7)
txt(ax, COL1_CX, 5.22, 'Normalise  (ImageNet μ/σ)', size=8.5, weight='bold', color=C['data_b'])
txt(ax, COL1_CX, 4.97, 'mean=[0.485,0.456,0.406]  std=[0.229,0.224,0.225]', size=7.5, color=C['subtext'])

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 2 — MODEL ARCHITECTURE  (x: 5.9 → 12.9)
# ══════════════════════════════════════════════════════════════════════════════
COL2_X  = 6.0
COL2_W  = 7.0
COL2_CX = COL2_X + COL2_W / 2

section_header(ax, COL2_X, 13.0, COL2_W, '② MODEL ARCHITECTURE', C['model_b'], C['model'])

# Input tensor
box(ax, COL2_X+1.5, 12.0, COL2_W-3.0, 0.55, C['model'], C['model_b'], radius=0.2)
txt(ax, COL2_CX, 12.29, 'Input Tensor  224 × 224 × 3', size=9, weight='bold', color=C['model_b'])

arrow(ax, COL2_CX, 12.0, COL2_CX, 11.45)

# EfficientNet-B2 backbone
box(ax, COL2_X+0.3, 9.75, COL2_W-0.6, 1.55, C['model'], C['model_b'], radius=0.2)
txt(ax, COL2_CX, 11.05, 'EfficientNet-B2 Backbone', size=10, weight='bold', color=C['model_b'])
txt(ax, COL2_CX, 10.75, 'Pretrained on ImageNet  ·  9.1 M parameters', size=8, color=C['subtext'])

# MBConv blocks
layers = [
    ('Stem Conv2d', '32 ch'),('MBConv1', '16 ch'),('MBConv6×2', '24 ch'),
    ('MBConv6×3', '48 ch'),('MBConv6×3', '88 ch'),('MBConv6×4', '120 ch'),
    ('MBConv6×4', '208 ch'),('MBConv6×1', '352 ch'),('Head Conv2d', '1408 ch'),
]
cols_n = 3
row_h  = 0.26
col_w  = (COL2_W - 0.8) / cols_n
for i, (name, ch) in enumerate(layers):
    r, c = divmod(i, cols_n)
    xi = COL2_X + 0.4 + c * col_w
    yi = 10.65 - r * row_h
    box(ax, xi, yi-0.2, col_w-0.1, 0.22, '#0a2a18', C['model_b'], radius=0.08, lw=0.8)
    txt(ax, xi+col_w*0.5-0.05, yi-0.09, f'{name}  {ch}', size=6.5, color=C['model_b'])

arrow(ax, COL2_CX, 9.75, COL2_CX, 9.2)

# AvgPool
box(ax, COL2_X+1.5, 8.65, COL2_W-3.0, 0.42, '#0a2a18', C['model_b'], radius=0.15)
txt(ax, COL2_CX, 8.88, 'AdaptiveAvgPool2d  →  1×1×1408', size=8.5, weight='bold', color=C['model_b'])

arrow(ax, COL2_CX, 8.65, COL2_CX, 8.1)

# Custom head
box(ax, COL2_X+0.3, 6.2, COL2_W-0.6, 1.75, C['model'], C['model_b'], radius=0.2)
txt(ax, COL2_CX, 7.7, 'Custom Classifier Head', size=10, weight='bold', color=C['model_b'])
head_steps = [
    ('Dropout  p=0.3', '#1a3d20'),
    ('Linear  1408 → 512', '#1a3d20'),
    ('ReLU', '#1a3d20'),
    ('Dropout  p=0.3', '#1a3d20'),
    ('Linear  512 → 7', '#22522a'),
]
step_h = 0.26
for i, (label, fc) in enumerate(head_steps):
    yi = 7.38 - i * step_h
    box(ax, COL2_X+1.0, yi-0.18, COL2_W-2.0, 0.23, fc, C['model_b'], radius=0.08, lw=0.8)
    txt(ax, COL2_CX, yi-0.065, label, size=8, color=C['model_b'])

arrow(ax, COL2_CX, 6.2, COL2_CX, 5.65)

# Output — 7 emotion nodes
out_y  = 5.2
node_w = 0.82
gap    = (COL2_W - len(EMOTIONS)*node_w) / (len(EMOTIONS)+1)
for i, (em, col) in enumerate(zip(EMOTIONS, C['emotion'])):
    xi = COL2_X + gap + i*(node_w+gap)
    box(ax, xi, out_y-0.24, node_w, 0.42, col+'33', col, radius=0.15, lw=1.5)
    txt(ax, xi+node_w/2, out_y-0.03, em, size=7.5, weight='bold', color=col)

txt(ax, COL2_CX, 4.72, 'Softmax  →  probability distribution over 7 classes', size=8, color=C['subtext'])

# Freeze/unfreeze note
box(ax, COL2_X+0.3, 4.05, COL2_W-0.6, 0.52, '#0a200a', C['model_b'], radius=0.15, lw=1, alpha=0.8)
txt(ax, COL2_CX, 4.35, 'Freeze strategy: backbone frozen for epochs 1–5, fully trainable from epoch 6', size=7.8, color=C['subtext'])
txt(ax, COL2_CX, 4.12, '(avoids destroying pretrained features during early classifier training)', size=7, color=C['subtext'], style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 3 — TRAINING  (x: 13.3 → 18.3)
# ══════════════════════════════════════════════════════════════════════════════
COL3_X  = 13.3
COL3_W  = 4.9
COL3_CX = COL3_X + COL3_W / 2

section_header(ax, COL3_X, 13.0, COL3_W, '③ TRAINING', C['train_b'], C['train'])

train_blocks = [
    ('Loss Function', ['LabelSmoothingCrossEntropy', 'smoothing = 0.1', 'Class weights (inv-freq, max 5×)']),
    ('Optimiser', ['Adam', 'lr = 0.001', 'weight_decay = 1e-4']),
    ('LR Scheduler', ['CosineAnnealingLR', 'T_max = 20 epochs', 'η_min = 1e-6']),
    ('Checkpointing', ['Save best_model.pth', 'every 10 epochs → checkpoint_epoch_N.pth', f'Monitored metric: val accuracy']),
]

y_cursor = 12.35
for title, lines in train_blocks:
    h = 0.32 + len(lines)*0.24
    box(ax, COL3_X, y_cursor-h, COL3_W, h, C['train'], C['train_b'], radius=0.2)
    txt(ax, COL3_CX, y_cursor-0.22, title, size=9, weight='bold', color=C['train_b'])
    for j, line in enumerate(lines):
        txt(ax, COL3_X+0.35, y_cursor-0.46-j*0.24, f'▸ {line}', size=7.5, color=C['subtext'], ha='left')
    y_cursor -= h + 0.2

# Training loop diagram
loop_y = y_cursor - 0.15
box(ax, COL3_X, loop_y-1.55, COL3_W, 1.55, C['train'], C['train_b'], radius=0.2)
txt(ax, COL3_CX, loop_y-0.22, 'Training Loop  (20 epochs)', size=9, weight='bold', color=C['train_b'])
loop_steps = ['Forward pass', 'Compute loss', 'Backward pass', 'optimizer.step()', 'scheduler.step()']
for j, s in enumerate(loop_steps):
    txt(ax, COL3_X+0.35, loop_y-0.46-j*0.22, f'▸ {s}', size=7.5, color=C['subtext'], ha='left')

# Duration note
box(ax, COL3_X, 4.25, COL3_W, 0.65, '#1a0a2e', C['train_b'], radius=0.15, lw=1, alpha=0.8)
txt(ax, COL3_CX, 4.62, 'Apple M1 8 GB  —  ~6–8 min/epoch', size=8.5, weight='bold', color=C['train_b'])
txt(ax, COL3_CX, 4.38, '20 epochs ≈ 2–2.5 hours total', size=7.8, color=C['subtext'])

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 4 — EVALUATION + DEMO  (x: 18.55 → 23.65)
# ══════════════════════════════════════════════════════════════════════════════
COL4_X  = 18.55
COL4_W  = 5.1
COL4_CX = COL4_X + COL4_W / 2

section_header(ax, COL4_X, 13.0, COL4_W, '④ EVALUATION & DEMO', C['eval_b'], C['eval'])

# Cross-dataset eval
box(ax, COL4_X, 10.5, COL4_W, 2.35, C['eval'], C['eval_b'], radius=0.2)
txt(ax, COL4_CX, 12.6, 'Cross-Dataset Evaluation', size=9.5, weight='bold', color=C['eval_b'])
results = [
    ('FER2013  (test)', '~60%', '~0.60'),
    ('CK+48',           '~60%', '~0.60'),
    ('JAFFE',           'weak', '—'),
]
txt(ax, COL4_X+0.4,  12.28, 'Dataset', size=7.5, color=C['subtext'], weight='bold', ha='left')
txt(ax, COL4_X+2.55, 12.28, 'Acc', size=7.5, color=C['subtext'], weight='bold', ha='center')
txt(ax, COL4_X+3.8,  12.28, 'F1', size=7.5,  color=C['subtext'], weight='bold', ha='center')
ax.axhline(12.12, xmin=(COL4_X)/24, xmax=(COL4_X+COL4_W)/24,
           color=C['eval_b'], lw=0.6, alpha=0.5, zorder=4)
for i,(ds,acc,f1) in enumerate(results):
    yi = 11.88 - i*0.42
    txt(ax, COL4_X+0.4,  yi, ds,  size=8,   color=C['text'],    ha='left')
    txt(ax, COL4_X+2.55, yi, acc, size=8,   color=C['eval_b'],  ha='center')
    txt(ax, COL4_X+3.8,  yi, f1,  size=8,   color=C['eval_b'],  ha='center')

txt(ax, COL4_X+0.35, 10.62,
    'JAFFE weak: domain gap\n(Japanese female, lab)', size=7, color=C['subtext'],
    ha='left', style='italic')

# Per-class metrics
box(ax, COL4_X, 8.45, COL4_W, 1.85, C['eval'], C['eval_b'], radius=0.2)
txt(ax, COL4_CX, 10.08, 'Per-Class F1  (FER2013)', size=9, weight='bold', color=C['eval_b'])
class_f1 = [0.62, 0.45, 0.48, 0.78, 0.55, 0.72, 0.60]  # approx achieved
bar_w = (COL4_W - 0.5) / 7
for i, (em, f, col) in enumerate(zip(EMOTIONS, class_f1, C['emotion'])):
    xi   = COL4_X + 0.25 + i*bar_w
    bh   = f * 1.3
    box(ax, xi, 8.55, bar_w-0.08, bh, col+'44', col, radius=0.06, lw=0.8)
    txt(ax, xi+bar_w*0.46, 8.55+bh+0.08, f'{f:.2f}', size=6, color=col)
    txt(ax, xi+bar_w*0.46, 8.58, em[:3], size=6, color=C['subtext'])

# Gradio demo
box(ax, COL4_X, 5.8, COL4_W, 2.45, C['demo'], C['demo_b'], radius=0.2)
txt(ax, COL4_CX, 8.0, 'Gradio Webcam Demo  (app.py)', size=9.5, weight='bold', color=C['demo_b'])
demo_steps = [
    ('Webcam / Upload', 'browser input panel'),
    ('Haar cascade', 'face detection + padding'),
    ('Preprocess', 'crop → 224×224 → normalise'),
    ('Inference', 'EfficientNet-B2 on MPS'),
    ('Output', 'annotated image + bar chart'),
    ('Auto-trigger', 'cam.change() → instant result'),
]
for i, (k, v) in enumerate(demo_steps):
    yi = 7.72 - i*0.33
    txt(ax, COL4_X+0.4,  yi, f'▸ {k}:', size=8, color=C['demo_b'],  ha='left', weight='bold')
    txt(ax, COL4_X+2.05, yi, v,          size=8, color=C['subtext'], ha='left')

txt(ax, COL4_CX, 5.93, 'http://127.0.0.1:7860', size=8, color=C['demo_b'],
    weight='bold', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS between columns
# ══════════════════════════════════════════════════════════════════════════════
# Col1 → Col2 (at DataLoader level)
arrow(ax, COL1_X+COL1_W, 7.12, COL2_X, 7.12, color=C['arrow_hi'], lw=2, mutation_scale=16)
txt(ax, (COL1_X+COL1_W+COL2_X)/2, 7.3, 'batches', size=7.5, color=C['subtext'])

# Col2 → Col3 (at output level, going right)
arrow(ax, COL2_X+COL2_W, 9.6, COL3_X, 9.6, color=C['arrow_hi'], lw=2, mutation_scale=16)
txt(ax, (COL2_X+COL2_W+COL3_X)/2, 9.78, 'model', size=7.5, color=C['subtext'])

# Col3 → Col4 (checkpoint → eval)
arrow(ax, COL3_X+COL3_W, 9.6, COL4_X, 9.6, color=C['arrow_hi'], lw=2, mutation_scale=16)
txt(ax, (COL3_X+COL3_W+COL4_X)/2, 9.78, 'checkpoint', size=7.5, color=C['subtext'])

# Col3 → Col4 (demo, lower)
arrow(ax, COL3_X+COL3_W, 4.57, COL4_X, 7.02, color=C['demo_b'], lw=1.5,
      mutation_scale=14)
txt(ax, (COL3_X+COL3_W+COL4_X)/2-0.1, 5.7, 'best_model\n.pth', size=7,
    color=C['demo_b'], ha='center', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM — file map
# ══════════════════════════════════════════════════════════════════════════════
ax.axhline(3.7, color=C['border'], lw=0.8, zorder=1)
txt(ax, 12, 3.45, 'Key Source Files', size=9, weight='bold', color=C['accent'])

files = [
    ('config/config.yaml',              C['data_b'],    'all hyperparams & paths'),
    ('src/data/dataset_loader.py',      C['data_b'],    'FER2013 / CK+ / JAFFE loaders'),
    ('src/models/efficientnet_model.py',C['model_b'],   'EfficientNetFER + EnsembleFER'),
    ('src/models/trainer.py',           C['train_b'],   'FERTrainer — train / val / checkpoint'),
    ('src/evaluation/cross_dataset_eval.py', C['eval_b'],'CrossDatasetEvaluator'),
    ('app.py',                          C['demo_b'],    'Gradio webcam demo'),
    ('main.py',                         C['accent'],    'CLI entry point (train / evaluate)'),
]

total_w  = 23.3
item_w   = total_w / len(files)
y_file   = 3.1
for i, (path, col, desc) in enumerate(files):
    xi = 0.35 + i * item_w
    box(ax, xi, y_file-0.5, item_w-0.12, 0.85, col+'18', col, radius=0.15, lw=1.2)
    txt(ax, xi+item_w/2-0.06, y_file-0.07, path.split('/')[-1],
        size=7.8, weight='bold', color=col)
    short = desc[:28]+'…' if len(desc)>28 else desc
    txt(ax, xi+item_w/2-0.06, y_file-0.32, short, size=6.8, color=C['subtext'])
    txt(ax, xi+item_w/2-0.06, y_file-0.52+0.08,
        '/'.join(path.split('/')[:-1]) or '.', size=6, color=col, alpha=0.6, style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
txt(ax, 12, 0.25,
    'Advanced FER System  ·  EfficientNet-B2  ·  PyTorch + MPS  ·  Gradio 6.x  ·  April 2026',
    size=7.5, color=C['subtext'])

import os
for out in ['outputs/figures/architecture.png', 'docs/architecture.png']:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    print(f'Saved → {out}')
plt.close(fig)

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Paths
root = Path("d:/New26-27/diabetes-readmission-prediction")
fig_dir = root / "reports/figures"
eda_dir = fig_dir / "eda"

# Load source images
img_eda_target = Image.open(eda_dir / "eda_plot_1_cell_5.png") # (1180, 460)
img_eda_enc = Image.open(eda_dir / "eda_plot_3_cell_9.png")     # (940, 460)
img_roc = Image.open(fig_dir / "roc_curve.png")                 # (720, 600)
img_pr = Image.open(fig_dir / "pr_curve.png")                   # (720, 600)
img_cm = Image.open(fig_dir / "confusion_matrix.png")           # (600, 480)
img_shap = Image.open(fig_dir / "shap_summary.png")             # (1080, 720)

# Also rename/copy the top individual plots for easy sharing
img_eda_target.save(fig_dir / "01_before_target_imbalance.png")
img_shap.save(fig_dir / "03_after_shap_explainability.png")

# Combine ROC and PR into one side-by-side performance image
roc_pr_w = img_roc.width + img_pr.width
roc_pr_h = max(img_roc.height, img_pr.height)
roc_pr_comb = Image.new("RGB", (roc_pr_w, roc_pr_h), (255, 255, 255))
roc_pr_comb.paste(img_roc, (0, 0))
roc_pr_comb.paste(img_pr, (img_roc.width, 0))
roc_pr_comb.save(fig_dir / "02_after_model_performance.png")

# Now build the Comprehensive Master Dashboard (Canvas: 2400 x 1700, clean dark-navy header & crisp cards)
canvas_w = 2400
canvas_h = 1750
dashboard = Image.new("RGB", (canvas_w, canvas_h), (244, 246, 249)) # Light modern slate background
draw = ImageDraw.Draw(dashboard)

# Try default or standard fonts
try:
    font_title = ImageFont.truetype("arial.ttf", 46)
    font_subtitle = ImageFont.truetype("arial.ttf", 26)
    font_sec = ImageFont.truetype("arialbd.ttf", 32)
    font_badge = ImageFont.truetype("arialbd.ttf", 20)
    font_card_title = ImageFont.truetype("arialbd.ttf", 24)
    font_footer = ImageFont.truetype("arial.ttf", 22)
except:
    font_title = ImageFont.load_default()
    font_subtitle = font_title
    font_sec = font_title
    font_badge = font_title
    font_card_title = font_title
    font_footer = font_title

# 1. Header Banner (Dark Medical Navy)
draw.rectangle([(0, 0), (canvas_w, 140)], fill=(16, 37, 66)) # #102542

draw.text((60, 25), "HOSPITAL READMISSION PREDICTION — END-TO-END CLINICAL ML PIPELINE", fill=(255, 255, 255), font=font_title)
draw.text((60, 85), "UCI Diabetes 130-US Hospitals (101,766 Patients) | LightGBM + Linear SHAP + ONNX | PAEP Program", fill=(144, 194, 249), font=font_subtitle)

# 2. Section 1 Badge: BEFORE TRAINING (EDA & Imbalance)
draw.rounded_rectangle([(60, 165), (560, 215)], radius=10, fill=(217, 83, 79)) # Red badge
draw.text((80, 175), "PHASE 1: BEFORE TRAINING (EDA)", fill=(255, 255, 255), font=font_badge)

# Section 1 Cards
# Card 1: Target Imbalance
card1_x, card1_y, card1_w, card1_h = 60, 230, 1120, 580
draw.rounded_rectangle([(card1_x, card1_y), (card1_x + card1_w, card1_y + card1_h)], radius=12, fill=(255, 255, 255), outline=(218, 224, 233), width=2)
draw.text((card1_x + 30, card1_y + 20), "1. Class Imbalance & Readmission Target (<30d vs >30d vs No)", fill=(26, 43, 76), font=font_card_title)
# Resize and paste img_eda_target
eda_tgt_resized = img_eda_target.resize((1060, 480), Image.Resampling.LANCZOS)
dashboard.paste(eda_tgt_resized, (card1_x + 30, card1_y + 70))

# Card 2: Encounters per patient
card2_x, card2_y, card2_w, card2_h = 1220, 230, 1120, 580
draw.rounded_rectangle([(card2_x, card2_y), (card2_x + card2_w, card2_y + card2_h)], radius=12, fill=(255, 255, 255), outline=(218, 224, 233), width=2)
draw.text((card2_x + 30, card2_y + 20), "2. Patient Visit Frequencies — Preventing Data Leakage via GroupSplit", fill=(26, 43, 76), font=font_card_title)
eda_enc_resized = img_eda_enc.resize((1060, 480), Image.Resampling.LANCZOS)
dashboard.paste(eda_enc_resized, (card2_x + 30, card2_y + 70))


# 3. Section 2 Badge: AFTER TRAINING (Evaluation & Explainability)
draw.rounded_rectangle([(60, 835), (630, 885)], radius=10, fill=(40, 167, 69)) # Green badge
draw.text((80, 845), "PHASE 2: AFTER TRAINING (EVAL & XAI)", fill=(255, 255, 255), font=font_badge)

# Bottom Row Cards:
# Card 3: Model Comparison (PR-Curve & ROC)
card3_x, card3_y, card3_w, card3_h = 60, 900, 720, 740
draw.rounded_rectangle([(card3_x, card3_y), (card3_x + card3_w, card3_y + card3_h)], radius=12, fill=(255, 255, 255), outline=(218, 224, 233), width=2)
draw.text((card3_x + 25, card3_y + 20), "3. Model Comparison (PR-AUC 2x Baseline)", fill=(26, 43, 76), font=font_card_title)
pr_resized = img_pr.resize((670, 640), Image.Resampling.LANCZOS)
dashboard.paste(pr_resized, (card3_x + 25, card3_y + 70))

# Card 4: Confusion Matrix at Clinical Threshold
card4_x, card4_y, card4_w, card4_h = 810, 900, 680, 740
draw.rounded_rectangle([(card4_x, card4_y), (card4_x + card4_w, card4_y + card4_h)], radius=12, fill=(255, 255, 255), outline=(218, 224, 233), width=2)
draw.text((card4_x + 25, card4_y + 20), "4. Clinical Decision Threshold (0.651)", fill=(26, 43, 76), font=font_card_title)
cm_resized = img_cm.resize((630, 620), Image.Resampling.LANCZOS)
dashboard.paste(cm_resized, (card4_x + 25, card4_y + 80))

# Card 5: SHAP Explainability Beeswarm
card5_x, card5_y, card5_w, card5_h = 1520, 900, 820, 740
draw.rounded_rectangle([(card5_x, card5_y), (card5_x + card5_w, card5_y + card5_h)], radius=12, fill=(255, 255, 255), outline=(218, 224, 233), width=2)
draw.text((card5_x + 25, card5_y + 20), "5. Explainable AI — SHAP Clinical Risk Drivers", fill=(26, 43, 76), font=font_card_title)
shap_resized = img_shap.resize((780, 640), Image.Resampling.LANCZOS)
dashboard.paste(shap_resized, (card5_x + 20, card5_y + 70))

# Save Dashboard
out_dashboard = fig_dir / "pipeline_overview_dashboard.png"
dashboard.save(out_dashboard, quality=95)
print(f"Successfully generated Master Dashboard: {out_dashboard}")

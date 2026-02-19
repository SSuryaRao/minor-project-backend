from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

# Create presentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# Add blank slide
blank_slide_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_slide_layout)

# Add light gray background
background = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.6), Inches(0.8),
    Inches(9.4), Inches(6.2)
)
background.fill.solid()
background.fill.fore_color.rgb = RGBColor(245, 245, 245)
background.line.width = 0

# Add orange header corner
header_corner = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0), Inches(0),
    Inches(0.6), Inches(0.8)
)
header_corner.fill.solid()
header_corner.fill.fore_color.rgb = RGBColor(255, 127, 39)
header_corner.line.width = 0

# Add white header bar
header_bar = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.6), Inches(0),
    Inches(9.4), Inches(0.8)
)
header_bar.fill.solid()
header_bar.fill.fore_color.rgb = RGBColor(255, 255, 255)
header_bar.line.width = 0

# Add header text
header_text = slide.shapes.add_textbox(
    Inches(0.7), Inches(0.15),
    Inches(6.5), Inches(0.5)
)
tf = header_text.text_frame
tf.text = "B.TECH MINOR PROJECT PRESENTATION 2025-26"
p = tf.paragraphs[0]
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = RGBColor(255, 127, 39)

# Add NIST logo box
logo_box = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(8.7), Inches(0.1),
    Inches(1.2), Inches(0.6)
)
logo_box.fill.solid()
logo_box.fill.fore_color.rgb = RGBColor(47, 84, 150)
logo_box.line.width = 0

# Add logo circle
logo_circle = slide.shapes.add_shape(
    9,  # Oval
    Inches(8.82), Inches(0.18),
    Inches(0.22), Inches(0.22)
)
logo_circle.fill.solid()
logo_circle.fill.fore_color.rgb = RGBColor(255, 255, 255)
logo_circle.line.width = Pt(1.5)
logo_circle.line.color.rgb = RGBColor(255, 255, 255)

# Add NIST text
logo_text = slide.shapes.add_textbox(
    Inches(9.1), Inches(0.2),
    Inches(0.7), Inches(0.4)
)
tf = logo_text.text_frame
tf.text = "nist"
p = tf.paragraphs[0]
p.font.size = Pt(24)
p.font.bold = True
p.font.color.rgb = RGBColor(255, 255, 255)

# Add NIST UNIVERSITY sidebar
sidebar = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0), Inches(0.8),
    Inches(0.6), Inches(6.2)
)
sidebar.fill.solid()
sidebar.fill.fore_color.rgb = RGBColor(47, 84, 150)
sidebar.line.width = 0

# Add NIST UNIVERSITY text (rotated effect with individual letters)
sidebar_text = slide.shapes.add_textbox(
    Inches(0.08), Inches(2.5),
    Inches(0.45), Inches(3)
)
tf = sidebar_text.text_frame
tf.text = "NIST UNIVERSITY"
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = RGBColor(255, 255, 255)
p.alignment = PP_ALIGN.CENTER

# Add METHODOLOGY title
title_box = slide.shapes.add_textbox(
    Inches(0.7), Inches(0.85),
    Inches(9), Inches(0.6)
)
tf = title_box.text_frame
tf.text = "METHODOLOGY"
p = tf.paragraphs[0]
p.font.size = Pt(36)
p.font.bold = True
p.font.color.rgb = RGBColor(47, 84, 150)

# Add horizontal red line under title
line = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.7), Inches(1.48),
    Inches(9.2), Inches(0.03)
)
line.fill.solid()
line.fill.fore_color.rgb = RGBColor(200, 0, 0)
line.line.width = 0

# Add white box for Architecture
arch_bg = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.8), Inches(1.65),
    Inches(3.8), Inches(4.5)
)
arch_bg.fill.solid()
arch_bg.fill.fore_color.rgb = RGBColor(255, 255, 255)
arch_bg.line.width = Pt(1)
arch_bg.line.color.rgb = RGBColor(220, 220, 220)

# Add Architecture title
arch_title = slide.shapes.add_textbox(
    Inches(0.9), Inches(1.75),
    Inches(3.6), Inches(0.35)
)
tf = arch_title.text_frame
tf.text = "5-Layer System Architecture"
p = tf.paragraphs[0]
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(0, 0, 0)

# Layer 1 - Blue bar
layer1_bar = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.95), Inches(2.2),
    Inches(0.05), Inches(0.55)
)
layer1_bar.fill.solid()
layer1_bar.fill.fore_color.rgb = RGBColor(66, 133, 244)
layer1_bar.line.width = 0

# Layer 1 icon and text
layer1_icon = slide.shapes.add_textbox(
    Inches(1.05), Inches(2.22),
    Inches(0.25), Inches(0.2)
)
tf = layer1_icon.text_frame
tf.text = "≡"
p = tf.paragraphs[0]
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(66, 133, 244)

layer1_title = slide.shapes.add_textbox(
    Inches(1.3), Inches(2.22),
    Inches(3), Inches(0.2)
)
tf = layer1_title.text_frame
tf.text = "Layer 1: Data Collection"
p = tf.paragraphs[0]
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = RGBColor(66, 133, 244)

layer1_desc = slide.shapes.add_textbox(
    Inches(1.05), Inches(2.45),
    Inches(3.3), Inches(0.3)
)
tf = layer1_desc.text_frame
tf.word_wrap = True
tf.text = "Historical Sales, Store Data, Holidays, Oil Prices, Transactions, Kaggle Store Sales, 54 Stores, 33 Families"
p = tf.paragraphs[0]
p.font.size = Pt(7.5)
p.font.color.rgb = RGBColor(100, 100, 100)

# Layer 2 - Green bar
layer2_bar = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.95), Inches(2.85),
    Inches(0.05), Inches(0.55)
)
layer2_bar.fill.solid()
layer2_bar.fill.fore_color.rgb = RGBColor(76, 175, 80)
layer2_bar.line.width = 0

# Layer 2 icon and text
layer2_icon = slide.shapes.add_textbox(
    Inches(1.05), Inches(2.87),
    Inches(0.25), Inches(0.2)
)
tf = layer2_icon.text_frame
tf.text = "◉"
p = tf.paragraphs[0]
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = RGBColor(76, 175, 80)

layer2_title = slide.shapes.add_textbox(
    Inches(1.3), Inches(2.87),
    Inches(3), Inches(0.2)
)
tf = layer2_title.text_frame
tf.text = "Layer 2: Data Preprocessing"
p = tf.paragraphs[0]
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = RGBColor(76, 175, 80)

layer2_desc = slide.shapes.add_textbox(
    Inches(1.05), Inches(3.1),
    Inches(3.3), Inches(0.3)
)
tf = layer2_desc.text_frame
tf.word_wrap = True
tf.text = "Cleaning, Integration, Feature Engineering, Mission Imputed Vs. Rolling averages, holiday proximity, promotions"
p = tf.paragraphs[0]
p.font.size = Pt(7.5)
p.font.color.rgb = RGBColor(100, 100, 100)

# Layer 3 - Purple bar
layer3_bar = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.95), Inches(3.5),
    Inches(0.05), Inches(0.55)
)
layer3_bar.fill.solid()
layer3_bar.fill.fore_color.rgb = RGBColor(156, 39, 176)
layer3_bar.line.width = 0

# Layer 3 icon and text
layer3_icon = slide.shapes.add_textbox(
    Inches(1.05), Inches(3.52),
    Inches(0.25), Inches(0.2)
)
tf = layer3_icon.text_frame
tf.text = "⚬"
p = tf.paragraphs[0]
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = RGBColor(156, 39, 176)

layer3_title = slide.shapes.add_textbox(
    Inches(1.3), Inches(3.52),
    Inches(3), Inches(0.2)
)
tf = layer3_title.text_frame
tf.text = "Layer 3: ML Models"
p = tf.paragraphs[0]
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = RGBColor(156, 39, 176)

layer3_desc = slide.shapes.add_textbox(
    Inches(1.05), Inches(3.75),
    Inches(3.3), Inches(0.3)
)
tf = layer3_desc.text_frame
tf.word_wrap = True
tf.text = "Prophet, ARIMA, LSTM (4-layer)\nHyperparameter tuning, cross-validation"
p = tf.paragraphs[0]
p.font.size = Pt(7.5)
p.font.color.rgb = RGBColor(100, 100, 100)

# Layer 4 - Red bar
layer4_bar = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.95), Inches(4.15),
    Inches(0.05), Inches(0.55)
)
layer4_bar.fill.solid()
layer4_bar.fill.fore_color.rgb = RGBColor(244, 67, 54)
layer4_bar.line.width = 0

# Layer 4 icon and text
layer4_icon = slide.shapes.add_textbox(
    Inches(1.05), Inches(4.17),
    Inches(0.25), Inches(0.2)
)
tf = layer4_icon.text_frame
tf.text = "📊"
p = tf.paragraphs[0]
p.font.size = Pt(10)

layer4_title = slide.shapes.add_textbox(
    Inches(1.3), Inches(4.17),
    Inches(3), Inches(0.2)
)
tf = layer4_title.text_frame
tf.text = "Layer 4: Inventory Optimization"
p = tf.paragraphs[0]
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = RGBColor(244, 67, 54)

layer4_desc = slide.shapes.add_textbox(
    Inches(1.05), Inches(4.4),
    Inches(3.3), Inches(0.3)
)
tf = layer4_desc.text_frame
tf.word_wrap = True
tf.text = "Reorder Point, Safety Stock, Order Quantity, Alerts\nStreamlit + Plotly for user-friendly interface"
p = tf.paragraphs[0]
p.font.size = Pt(7.5)
p.font.color.rgb = RGBColor(100, 100, 100)

# Layer 5 - Orange bar (NEW LAYER)
layer5_bar = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0.95), Inches(4.8),
    Inches(0.05), Inches(0.55)
)
layer5_bar.fill.solid()
layer5_bar.fill.fore_color.rgb = RGBColor(255, 152, 0)
layer5_bar.line.width = 0

# Layer 5 icon and text
layer5_icon = slide.shapes.add_textbox(
    Inches(1.05), Inches(4.82),
    Inches(0.25), Inches(0.2)
)
tf = layer5_icon.text_frame
tf.text = "📱"
p = tf.paragraphs[0]
p.font.size = Pt(10)

layer5_title = slide.shapes.add_textbox(
    Inches(1.3), Inches(4.82),
    Inches(3), Inches(0.2)
)
tf = layer5_title.text_frame
tf.text = "Layer 5: Deployment & Monitoring"
p = tf.paragraphs[0]
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = RGBColor(255, 152, 0)

layer5_desc = slide.shapes.add_textbox(
    Inches(1.05), Inches(5.05),
    Inches(3.3), Inches(0.3)
)
tf = layer5_desc.text_frame
tf.word_wrap = True
tf.text = "Real-time dashboard, Performance metrics, Alert system\nContinuous model retraining and updates"
p = tf.paragraphs[0]
p.font.size = Pt(7.5)
p.font.color.rgb = RGBColor(100, 100, 100)

# Add center "Methodology" title
methodology_title = slide.shapes.add_textbox(
    Inches(4.8), Inches(1.8),
    Inches(2.2), Inches(0.5)
)
tf = methodology_title.text_frame
tf.text = "Methodology"
p = tf.paragraphs[0]
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(0, 0, 0)
p.alignment = PP_ALIGN.CENTER

# Add chart bars (blue)
chart_bar1 = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(4.9), Inches(3.1),
    Inches(0.25), Inches(0.9)
)
chart_bar1.fill.solid()
chart_bar1.fill.fore_color.rgb = RGBColor(66, 133, 244)
chart_bar1.line.width = 0

# Yellow bar
chart_bar2 = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(5.25), Inches(2.7),
    Inches(0.25), Inches(1.3)
)
chart_bar2.fill.solid()
chart_bar2.fill.fore_color.rgb = RGBColor(255, 193, 7)
chart_bar2.line.width = 0

# Red bar
chart_bar3 = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(5.6), Inches(2.3),
    Inches(0.25), Inches(1.7)
)
chart_bar3.fill.solid()
chart_bar3.fill.fore_color.rgb = RGBColor(244, 67, 54)
chart_bar3.line.width = 0

# Add trend line
# (simplified as a diagonal line shape)
trend_line = slide.shapes.add_textbox(
    Inches(4.85), Inches(2.2),
    Inches(1.2), Inches(0.4)
)
tf = trend_line.text_frame
tf.text = "📈"
p = tf.paragraphs[0]
p.font.size = Pt(40)

# Add magnifying glass
magnify = slide.shapes.add_shape(
    9,  # Oval
    Inches(5.7), Inches(3.2),
    Inches(0.5), Inches(0.5)
)
magnify.fill.solid()
magnify.fill.fore_color.rgb = RGBColor(255, 255, 255)
magnify.line.width = Pt(8)
magnify.line.color.rgb = RGBColor(244, 67, 54)

# Add bullet points on right
bullets_box = slide.shapes.add_textbox(
    Inches(7.1), Inches(1.7),
    Inches(2.7), Inches(1.8)
)
tf = bullets_box.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "• We collect daily sales data, weather information and calendar-based event data."
p.font.size = Pt(8.5)
p.font.color.rgb = RGBColor(0, 0, 0)
p.space_after = Pt(10)

p = tf.add_paragraph()
p.text = "• Data is preprocessed for leaning, grouping, and seasonal trend extraction."
p.font.size = Pt(8.5)
p.font.color.rgb = RGBColor(0, 0, 0)
p.space_after = Pt(10)

p = tf.add_paragraph()
p.text = "• Forecasting models such as ARIMA, Prophet, and LSTM are trained on historical data."
p.font.size = Pt(8.5)
p.font.color.rgb = RGBColor(0, 0, 0)

# Add settings gear icon
gear_box = slide.shapes.add_textbox(
    Inches(7.2), Inches(3.7),
    Inches(0.6), Inches(0.6)
)
tf = gear_box.text_frame
tf.text = "⚙"
p = tf.paragraphs[0]
p.font.size = Pt(48)
p.font.color.rgb = RGBColor(255, 193, 7)

# Add cloud icon with box
cloud_bg = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(7.9), Inches(3.7),
    Inches(0.7), Inches(0.5)
)
cloud_bg.fill.solid()
cloud_bg.fill.fore_color.rgb = RGBColor(100, 181, 246)
cloud_bg.line.width = 0

cloud_icon = slide.shapes.add_textbox(
    Inches(8.05), Inches(3.75),
    Inches(0.4), Inches(0.4)
)
tf = cloud_icon.text_frame
tf.text = "☁"
p = tf.paragraphs[0]
p.font.size = Pt(32)
p.font.color.rgb = RGBColor(255, 255, 255)

# Add server bars
server_box = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(7.9), Inches(4.3),
    Inches(1.8), Inches(0.35)
)
server_box.fill.solid()
server_box.fill.fore_color.rgb = RGBColor(66, 133, 244)
server_box.line.width = 0

# Server icons
server_dots = slide.shapes.add_textbox(
    Inches(8), Inches(4.35),
    Inches(1.6), Inches(0.25)
)
tf = server_dots.text_frame
tf.text = "▭  ▭  ▭"
p = tf.paragraphs[0]
p.font.size = Pt(14)
p.font.color.rgb = RGBColor(255, 255, 255)
p.alignment = PP_ALIGN.CENTER

# Add package box icon
package = slide.shapes.add_textbox(
    Inches(7.2), Inches(4.8),
    Inches(0.8), Inches(0.8)
)
tf = package.text_frame
tf.text = "📦"
p = tf.paragraphs[0]
p.font.size = Pt(60)

# Add chart icon
chart = slide.shapes.add_textbox(
    Inches(8.1), Inches(4.8),
    Inches(0.8), Inches(0.8)
)
tf = chart.text_frame
tf.text = "📊"
p = tf.paragraphs[0]
p.font.size = Pt(60)

# Add checkmark
check = slide.shapes.add_textbox(
    Inches(9), Inches(4.85),
    Inches(0.6), Inches(0.6)
)
tf = check.text_frame
tf.text = "✓"
p = tf.paragraphs[0]
p.font.size = Pt(48)
p.font.color.rgb = RGBColor(66, 133, 244)

# Add footer with student names
footer_bg = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(0), Inches(6.9),
    Inches(9.4), Inches(0.6)
)
footer_bg.fill.solid()
footer_bg.fill.fore_color.rgb = RGBColor(255, 255, 255)
footer_bg.line.width = 0

footer_text = slide.shapes.add_textbox(
    Inches(0.7), Inches(6.95),
    Inches(8.5), Inches(0.5)
)
tf = footer_text.text_frame
tf.text = "202220576  Sodi Surya Rao     202220577 Sunil Dalai     202212582 Suyam Mishra     202220519  Ankit Kumar"
p = tf.paragraphs[0]
p.font.size = Pt(10)
p.font.color.rgb = RGBColor(0, 0, 0)
p.alignment = PP_ALIGN.LEFT

# Page number box
page_num = slide.shapes.add_shape(
    1,  # Rectangle
    Inches(9.4), Inches(6.9),
    Inches(0.6), Inches(0.6)
)
page_num.fill.solid()
page_num.fill.fore_color.rgb = RGBColor(255, 127, 39)
page_num.line.width = 0

page_text = page_num.text_frame
page_text.text = "6"
page_text.paragraphs[0].font.size = Pt(22)
page_text.paragraphs[0].font.bold = True
page_text.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
page_text.paragraphs[0].alignment = PP_ALIGN.CENTER
page_text.vertical_anchor = MSO_ANCHOR.MIDDLE

# Save presentation
prs.save('E:\\Coding\\minorProject\\Methodology_Presentation.pptx')
print("Presentation created successfully with 5 layers!")

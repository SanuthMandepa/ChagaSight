import base64

SVG_DB = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>'
SVG_ECG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>'
SVG_IMG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>'
SVG_GEAR = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'
SVG_NETWORK = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>'
SVG_AUG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>'
SVG_PERSON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>'
SVG_SPLIT = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3h5v5"></path><path d="M8 3H3v5"></path><path d="M12 22v-8.3a4 4 0 0 0-1.17-2.83l-7-7"></path><path d="M15 9l6-6"></path></svg>'
SVG_PLUS = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>'

def get_b64(svg_str):
    return "data:image/svg+xml;base64," + base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')

class DrawioBuilder:
    def __init__(self):
        self.cells = []
        self.id_counter = 2

    def add_cell(self, value, x, y, w, h, style, parent="1", vertex="1", edge=None, source=None, target=None, waypoints=None):
        cell_id = str(self.id_counter)
        self.id_counter += 1
        
        geom = f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry">'
        if waypoints:
            geom += '<Array as="points">'
            for px, py in waypoints:
                geom += f'<mxPoint x="{px}" y="{py}" />'
            geom += '</Array>'
        geom += '</mxGeometry>'
        
        if edge == "1" and not waypoints:
            geom = f'<mxGeometry relative="1" as="geometry"/>'
            
        edge_attr = ' edge="1"' if edge else ''
        vertex_attr = ' vertex="1"' if vertex and not edge else ''
        src_attr = f' source="{source}"' if source else ''
        tgt_attr = f' target="{target}"' if target else ''
        
        # Escape XML entities in value
        value_esc = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
        
        cell = f'<mxCell id="{cell_id}" value="{value_esc}" style="{style}" parent="{parent}"{vertex_attr}{edge_attr}{src_attr}{tgt_attr}>{geom}</mxCell>'
        self.cells.append(cell)
        return cell_id

    def build(self):
        header = """<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="Electron" modified="2023-10-01T00:00:00.000Z" agent="Mozilla/5.0" version="21.6.8" type="device">
  <diagram id="diagram-id" name="ChagaSight">
    <mxGraphModel dx="2000" dy="2000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="3600" pageHeight="2000" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
"""
        footer = """
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
        return header + "\n".join(self.cells) + footer

builder = DrawioBuilder()

# Scale factor
S = 100

def box(x, y, w, h, fill, stroke, text="", title="", fs=12):
    style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;align=left;verticalAlign=top;spacingLeft=10;spacingTop=10;fontColor={stroke};fontSize={fs};fontStyle=1"
    val = f"{title}" if title else text
    return builder.add_cell(val, x*S, y*S, w*S, h*S, style)

def node(x, y, w, h, title, desc, icon_svg, fill, stroke, font_color="#ffffff"):
    icon_b64 = get_b64(icon_svg)
    html_val = f"""<div style="text-align:center;"><img src="{icon_b64}" width="28" height="28"/><br/><b style="font-size:13px;">{title}</b><br/><span style="font-size:11px;color:#eeeeee;">{desc}</span></div>"""
    style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontColor={font_color};strokeWidth=1"
    return builder.add_cell(html_val, x*S, y*S, w*S, h*S, style)

def block(x, y, w, h, html_val, fill, stroke, font_color="#ffffff", align="center"):
    style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontColor={font_color};strokeWidth=1;align={align}"
    return builder.add_cell(html_val, x*S, y*S, w*S, h*S, style)

def text(x, y, w, h, val, color="#000000", fs=12, bold=False):
    style = f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor={color};fontSize={fs};"
    if bold: style += "fontStyle=1;"
    return builder.add_cell(val, x*S, y*S, w*S, h*S, style)

def arrow(src, tgt, waypoints=None, color="#424242", lw=2):
    style = f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};strokeWidth={lw};endArrow=block;endFill=1;"
    if waypoints:
        wp_scaled = [(x*S, y*S) for x, y in waypoints]
        return builder.add_cell("", 0, 0, 0, 0, style, edge="1", source=src, target=tgt, waypoints=wp_scaled)
    else:
        return builder.add_cell("", 0, 0, 0, 0, style, edge="1", source=src, target=tgt)


# ==========================================
# DRAW.IO LAYOUT
# ==========================================
text(12, 0.2, 12, 1, "ChagaSight: Dual-Pathway Hybrid Architecture for Chagas Disease Detection", color="#0D47A1", fs=24, bold=True)

# A. Data Pipeline
box_A = box(0.2, 1, 8.3, 9.0, fill="#E8F0FE", stroke="#4285F4", title="A. Data Pipeline", fs=16)

box_dbs = box(0.4, 1.8, 3.2, 2.8, fill="#ffffff", stroke="#4285F4")
text(0.4, 1.8, 3.2, 0.4, "ECG Databases", color="#4285F4", fs=14, bold=True)
db1 = node(0.6, 2.3, 2.8, 0.65, "PTB-XL", "(~21,837 records)", SVG_DB, fill="#1A73E8", stroke="#0D47A1")
db2 = node(0.6, 3.1, 2.8, 0.65, "SaMi-Trop", "", SVG_DB, fill="#1A73E8", stroke="#0D47A1")
db3 = node(0.6, 3.9, 2.8, 0.65, "CODE-15%", "(soft labels: 0.2 / 0.8)", SVG_DB, fill="#5C6BC0", stroke="#3F51B5")

box_split = box(4.1, 1.8, 2.2, 2.3, fill="#ffffff", stroke="#4285F4")
text(4.1, 1.8, 2.2, 0.4, "Final Split", color="#4285F4", fs=14, bold=True)
split1 = node(4.25, 2.3, 1.9, 0.5, "Train", "", SVG_SPLIT, fill="#43A047", stroke="#2E7D32")
split2 = block(4.25, 2.9, 1.9, 0.5, "<b>Validation</b>", fill="#FB8C00", stroke="#E65100")
split3 = block(4.25, 3.5, 1.9, 0.5, "<b>Test</b> (held-out)", fill="#E53935", stroke="#C62828")

sampler = node(6.7, 2.3, 1.6, 1.2, "Weighted<br>Random<br>Sampler", "(5x pos weight)", SVG_GEAR, fill="#00897B", stroke="#00695C")

arrow(box_dbs, box_split)
arrow(split1, sampler)

# B. Preprocessing
box_B = box(0.4, 5.0, 8.0, 4.8, fill="#E8F5E9", stroke="#34A853", title="B. Data Preprocessing", fs=16)

raw = node(0.6, 5.8, 1.5, 0.9, "Raw 12-Lead", "ECG Signal", SVG_ECG, fill="#546E7A", stroke="#37474F")
resample = node(2.5, 5.8, 1.4, 0.9, "Resample", "500Hz (2D)<br>100Hz (1D)", SVG_GEAR, fill="#34A853", stroke="#1B5E20")
filter = node(4.2, 5.8, 1.5, 0.9, "Bandpass Filter", "0.5-40 Hz", SVG_GEAR, fill="#34A853", stroke="#1B5E20")
zscore = node(6.0, 5.8, 1.4, 0.9, "Z-Score", "Normalize per-lead", SVG_GEAR, fill="#34A853", stroke="#1B5E20")

arrow(raw, resample)
arrow(resample, filter)
arrow(filter, zscore)

# 2D / 1D split nodes
img_wct_html = f"""<div style="text-align:center;"><img src="{get_b64(SVG_IMG)}" width="24" height="24"/><br/><b>WCT Image Embedding (Kim et al. 2025)</b><br/><span style="font-size:11px;color:#eee;">Ch.0: RA-ref, Ch.1: LA-ref, Ch.2: LL-ref<br/>clip +/-3s  ->  uint8 [0, 255]</span></div>"""
img_wct = block(0.6, 7.8, 3.8, 1.6, img_wct_html, fill="#1565C0", stroke="#0D47A1")

sig_1d_html = f"""<div style="text-align:center;"><img src="{get_b64(SVG_ECG)}" width="24" height="24"/><br/><b>1D Signal Output</b><br/><span style="font-size:11px;color:#eee;">12 leads x 1000 samples<br/>100 Hz, 10 seconds (float32)</span></div>"""
sig_1d = block(5.0, 7.8, 2.8, 1.6, sig_1d_html, fill="#7B1FA2", stroke="#4A148C")

# arrows for splitting
arr_split_2d = builder.add_cell("", 0,0,0,0, f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#1565C0;strokeWidth=2;endArrow=block;endFill=1;", edge="1", source=zscore, target=img_wct, waypoints=[(6.7*S, 7.2*S), (2.5*S, 7.2*S)])
arr_split_1d = builder.add_cell("", 0,0,0,0, f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#7B1FA2;strokeWidth=2;endArrow=block;endFill=1;", edge="1", source=zscore, target=sig_1d, waypoints=[(6.7*S, 7.2*S), (6.4*S, 7.2*S)])

text(2.5, 7.3, 1.0, 0.4, "<b>2D Path</b>", color="#1565C0")
text(6.4, 7.3, 1.0, 0.4, "<b>1D Path</b>", color="#7B1FA2")

# C. Augmentation
box_C = box(8.8, 5.0, 3.8, 5.3, fill="#FFF3E0", stroke="#E65100", title="C. Data Augmentation (1D Signal Only)", fs=16)

def aug_row(y, title, desc):
    b = block(8.95, y, 1.5, 0.5, f"<b>{title}</b>", fill="#E65100", stroke="#BF360C")
    t = text(10.6, y, 1.8, 0.5, desc, color="#424242", fs=11)

aug_row(5.8, "Lead Mixup", "p=0.3, a=0.2, Beta interp.")
aug_row(6.6, "Powerline Noise", "p=0.5, SNR 15-30 dB, 50/60Hz")
aug_row(7.4, "Random Shift", "p=0.5, +/-100 samples")
aug_row(8.2, "Amplitude Scaling", "p=0.3, scale [0.8, 1.2]")
aug_row(9.0, "Baseline Wander", "p=0.2, 0.1-0.5 Hz")

# connect 1D to Aug
builder.add_cell("", 0,0,0,0, f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#7B1FA2;strokeWidth=2;endArrow=block;endFill=1;", edge="1", source=sig_1d, target=box_C)

# D. 2D-ViT Branch
box_D = box(13.0, 1.0, 14.0, 7.5, fill="#E3F2FD", stroke="#1565C0", title="D. 2D-ViT Branch (Contour Image Pathway)", fs=16)

in_2d = node(13.2, 2.5, 1.5, 1.3, "Input", "2D ECG Image<br>(B, 3, 24, 2048)", SVG_IMG, fill="#546E7A", stroke="#37474F")
mae = block(13.2, 4.0, 1.5, 0.7, "<b>MAE Pretrained</b><br>(optional)", fill="#78909C", stroke="#546E7A")
patch_2d = block(15.2, 2.2, 2.0, 1.7, "<b>PatchEmbed2D</b><hr/>Conv2d(3 &rarr; 768)<br>kernel = (8, 64)<br>stride = (8, 64)<br><br>3 x 32 = 96 patches", fill="#1565C0", stroke="#0D47A1")
pos_2d = block(17.6, 2.5, 1.4, 1.3, "<b>+ Pos Embed</b><hr/>(1, 96, 768)<br>trunc_normal<br>+ Dropout(0.1)", fill="#42A5F5", stroke="#1E88E5")

box_enc_2d = box(19.4, 2.0, 3.0, 3.6, fill="#BBDEFB", stroke="#1565C0", title="Transformer Encoder", fs=14)
mhsa_2d = node(19.55, 2.6, 2.7, 0.7, "Multi-Head Self-Attention", "(heads=12, head_dim=64)", SVG_NETWORK, fill="#0D47A1", stroke="#002171")
addnorm_2d = block(19.55, 3.5, 2.7, 0.65, "<b>Add & LayerNorm (Pre-LN)</b>", fill="#1565C0", stroke="#0D47A1")
mlp_2d = block(19.55, 4.3, 2.7, 0.7, "<b>MLP (768 &rarr; 3072 &rarr; 768)</b><br>GELU, Dropout(0.1)", fill="#0D47A1", stroke="#002171")

arrow(mhsa_2d, addnorm_2d)
arrow(addnorm_2d, mlp_2d)

ln_2d = block(23.2, 2.5, 1.2, 0.8, "<b>LayerNorm</b><br>(768)", fill="#42A5F5", stroke="#1E88E5")
box_aol_2d = box(24.8, 2.0, 2.0, 2.5, fill="#BBDEFB", stroke="#1565C0", title="AoL", fs=14)
aol_2d = block(24.95, 2.8, 1.7, 1.0, "<b>Aggregation of Layers</b><hr/>Mean-pool each layer<br>stack 12 layers<br>average", fill="#1565C0", stroke="#0D47A1")

# connections
builder.add_cell("", 0,0,0,0, "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#1565C0;strokeWidth=2;endArrow=block;", edge="1", source=img_wct, target=in_2d, waypoints=[(2.5*S, 10.0*S), (12.5*S, 10.0*S), (12.5*S, 3.15*S)])
arrow(in_2d, patch_2d, color="#1565C0")
arrow(patch_2d, pos_2d, color="#1565C0")
arrow(pos_2d, box_enc_2d, color="#1565C0")
arrow(box_enc_2d, ln_2d, color="#1565C0")
arrow(ln_2d, box_aol_2d, color="#1565C0")

# config
block(15.0, 7.5, 9.5, 0.6, "<b>2D-ViT Config:</b> DEPTH=12 | HEADS=12 | EMBED_DIM=768 | FFN_DIM=3072 | DROPOUT=0.1 | PATCH=(8,64) | Pre-LN", fill="#D2E3FC", stroke="#1565C0", font_color="#0D47A1")

text(22.6, 2.8, 0.5, 0.5, "<b>x12</b><br>layers", color="#1565C0", fs=14)
text(25.0, 4.7, 1.6, 0.4, "<b>f_img (B, 768)</b>", color="#1565C0", fs=14)

# E. 1D-ViT FM Branch
box_E = box(13.0, 9.0, 14.0, 8.2, fill="#F3E5F5", stroke="#7B1FA2", title="E. 1D-ViT FM Branch (Signal + Demographics Pathway)", fs=16)

in_1d = node(13.2, 10.5, 1.5, 1.3, "Input", "12-Lead ECG Signal<br>(B, 12, 1000)", SVG_ECG, fill="#546E7A", stroke="#37474F")
stmem = block(13.2, 12.0, 1.5, 0.7, "<b>ST-MEM Pretrained</b><br>(optional)", fill="#78909C", stroke="#546E7A")
patch_1d = block(15.2, 10.0, 2.0, 2.1, "<b>PatchEmbed1D</b><hr/>Conv1d(1 &rarr; 768)<br>kernel = 50, stride = 50<br>per lead<br><br>12 x 20 = 240 patches<hr/>+ Lead Embed (1,12,1,768)", fill="#7B1FA2", stroke="#4A148C")
pos_1d = block(17.6, 10.5, 1.4, 1.3, "<b>+ Pos Embed</b><hr/>(1, 240, 768)<br>trunc_normal<br>+ Dropout(0.1)", fill="#AB47BC", stroke="#7B1FA2")

box_enc_1d = box(19.4, 9.8, 3.0, 3.6, fill="#E1BEE7", stroke="#7B1FA2", title="Transformer Encoder", fs=14)
mhsa_1d = node(19.55, 10.4, 2.7, 0.7, "Multi-Head Self-Attention", "(heads=12, head_dim=64)", SVG_NETWORK, fill="#6A1B9A", stroke="#4A148C")
addnorm_1d = block(19.55, 11.3, 2.7, 0.65, "<b>Add & LayerNorm (Pre-LN)</b>", fill="#7B1FA2", stroke="#4A148C")
mlp_1d = block(19.55, 12.1, 2.7, 0.7, "<b>MLP (768 &rarr; 3072 &rarr; 768)</b><br>GELU, Dropout(0.1)", fill="#6A1B9A", stroke="#4A148C")
arrow(mhsa_1d, addnorm_1d)
arrow(addnorm_1d, mlp_1d)

ln_1d = block(23.2, 10.5, 1.2, 0.8, "<b>LayerNorm</b><br>(768)", fill="#AB47BC", stroke="#7B1FA2")
box_aol_1d = box(24.8, 9.8, 2.0, 2.5, fill="#E1BEE7", stroke="#7B1FA2", title="AoL", fs=14)
aol_1d = block(24.95, 10.6, 1.7, 1.0, "<b>Aggregation of Layers</b><hr/>Mean-pool each layer<br>stack 12 layers<br>average", fill="#7B1FA2", stroke="#4A148C")

# connections 1D
builder.add_cell("", 0,0,0,0, "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#7B1FA2;strokeWidth=2;endArrow=block;", edge="1", source=box_C, target=in_1d, waypoints=[(12.7*S, 7.65*S), (12.7*S, 11.15*S)])
arrow(in_1d, patch_1d, color="#7B1FA2")
arrow(patch_1d, pos_1d, color="#7B1FA2")
arrow(pos_1d, box_enc_1d, color="#7B1FA2")
arrow(box_enc_1d, ln_1d, color="#7B1FA2")
arrow(ln_1d, box_aol_1d, color="#7B1FA2")

text(22.6, 10.6, 0.5, 0.5, "<b>x12</b><br>layers", color="#7B1FA2", fs=14)
text(25.0, 12.5, 1.6, 0.4, "<b>f_sig (B, 768)</b>", color="#7B1FA2", fs=14)

# Demographics / FiLM
box_demo = box(13.2, 13.5, 6.0, 3.2, fill="#FCE4EC", stroke="#AD1457", title="Demographics Encoder (FiLM)", fs=14)
in_demo = node(13.4, 14.3, 1.2, 0.9, "Input", "Age, Sex (B, 2)", SVG_PERSON, fill="#880E4F", stroke="#4A0072")
mlp_demo = block(15.0, 14.0, 2.2, 1.7, "<b>MLP</b><hr/>Linear(2 &rarr; 256)<br>ReLU<br>Linear(256 &rarr; 256)<br>ReLU<br>Linear(256 &rarr; 1536)", fill="#AD1457", stroke="#880E4F")
split_demo = block(17.6, 14.0, 1.4, 1.7, "<b>Split</b><hr/>&gamma; (B,768)<br>&beta; (B,768)<hr/>init:<br>&gamma;=1, &beta;=0", fill="#C2185B", stroke="#880E4F")
film = node(20.0, 14.0, 2.0, 1.2, "FiLM Modality Weighting", "&gamma; &odot; f_sig + &beta;", SVG_GEAR, fill="#AD1457", stroke="#880E4F")

arrow(in_demo, mlp_demo, color="#AD1457")
arrow(mlp_demo, split_demo, color="#AD1457")
arrow(split_demo, film, color="#AD1457")

# connect AoL to FiLM
builder.add_cell("", 0,0,0,0, "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#7B1FA2;strokeWidth=2;endArrow=block;", edge="1", source=box_aol_1d, target=film, waypoints=[(25.8*S, 14.6*S)])

block(15.0, 16.5, 9.5, 0.6, "<b>1D-ViT FM Config:</b> DEPTH=12 | HEADS=12 | EMBED_DIM=768 | FFN_DIM=3072 | DROPOUT=0.1 | PATCH=50 | LEADS=12", fill="#E1BEE7", stroke="#7B1FA2", font_color="#4A148C")

# F. REPA Alignment
box_F = box(27.5, 3.0, 2.7, 4.0, fill="#E8F5E9", stroke="#2E7D32", title="F. REPA Alignment", fs=16)
repa_conv = block(27.7, 4.0, 2.3, 0.7, "<b>DepthwiseConv1d</b><br>(768 &rarr; 768, k=1, g=768)", fill="#2E7D32", stroke="#1B5E20")
repa_silu = block(27.7, 5.0, 2.3, 0.7, "<b>SiLU Activation</b>", fill="#388E3C", stroke="#1B5E20")
repa_lin = block(27.7, 6.0, 2.3, 0.7, "<b>Linear (768 &rarr; 768)</b>", fill="#2E7D32", stroke="#1B5E20")
arrow(repa_conv, repa_silu, color="#2E7D32")
arrow(repa_silu, repa_lin, color="#2E7D32")

# connect 2D to REPA
builder.add_cell("", 0,0,0,0, "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#1565C0;strokeWidth=2;endArrow=block;", edge="1", source=box_aol_2d, target=repa_conv, waypoints=[(28.85*S, 3.25*S)])

# G. Classification
box_G = box(27.5, 8.0, 8.2, 9.5, fill="#FFEBEE", stroke="#C62828", title="G. Fusion & Classification Head", fs=16)

concat = node(28.5, 8.8, 2.3, 0.9, "Concatenation", "[aligned_2d, f_sig_mod]", SVG_PLUS, fill="#00838F", stroke="#006064")
arrow(repa_lin, concat, color="#2E7D32", waypoints=[(28.85*S, 7.5*S), (29.65*S, 7.5*S)])
builder.add_cell("", 0,0,0,0, "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#7B1FA2;strokeWidth=2;endArrow=block;", edge="1", source=film, target=concat, waypoints=[(29.65*S, 14.6*S)])

l1 = block(29.0, 10.3, 2.5, 0.65, "<b>Linear (1536 &rarr; 512)</b>", fill="#C62828", stroke="#b71c1c")
r1 = block(29.0, 11.2, 2.5, 0.6, "<b>ReLU + Dropout(0.3)</b>", fill="#E53935", stroke="#c62828")
l2 = block(29.0, 12.1, 2.5, 0.6, "<b>Linear (512 &rarr; 256)</b>", fill="#C62828", stroke="#b71c1c")
r2 = block(29.0, 13.0, 2.5, 0.6, "<b>ReLU + Dropout(0.3)</b>", fill="#E53935", stroke="#c62828")
l3 = block(29.0, 13.9, 2.5, 0.6, "<b>Linear (256 &rarr; 1)</b>", fill="#C62828", stroke="#b71c1c")
logit = block(29.0, 14.8, 2.5, 0.55, "<b>Logit (B, 1)</b>", fill="#B71C1C", stroke="#880e4f")
sigm = block(29.0, 15.6, 2.5, 0.55, "<b>Sigmoid</b>", fill="#880E4F", stroke="#4a0072")

arrow(concat, l1, color="#C62828")
arrow(l1, r1, color="#C62828")
arrow(r1, l2, color="#C62828")
arrow(l2, r2, color="#C62828")
arrow(r2, l3, color="#C62828")
arrow(l3, logit, color="#C62828")
arrow(logit, sigm, color="#C62828")

pred = block(32.2, 13.0, 3.2, 3.5, "<div style='font-size:18px;'><b>Prediction</b><br><br>P(Chagas)<hr>&#9679; Positive<br>&#9675; Negative</div>", fill="#1B5E20", stroke="#1B5E20")
arrow(sigm, pred, color="#1B5E20")

io_summary = """<div style='text-align:left;padding:10px;'><b>Inputs:</b><ul style='margin-top:5px;'><li>Images: (B, 3, 24, 2048)</li><li>Signal: (B, 12, 1000)</li><li>Age: (B,) Sex: (B,)</li></ul><b>Output:</b><ul style='margin-top:5px;'><li>P(Chagas) &isin; [0, 1]</li></ul></div>"""
block(32.2, 8.8, 3.3, 3.0, io_summary, fill="#D2E3FC", stroke="#0D47A1", font_color="#0D47A1", align="left")

block(32.0, 17.0, 3.5, 0.6, "<b>Total Parameters: ~173M</b>", fill="#0D47A1", stroke="#002171")

# H. Training Strategy
box_H = box(0.2, 18.0, 12.4, 6.0, fill="#FFF8E1", stroke="#F57F17", title="H. Training Strategy", fs=16)

p1_txt = "<ul><li>Frozen: all 1D-ViT FM params (~85M)</li><li>Iterations: 2,000</li><li>Optimizer: AdamW (lr=2e-4)</li><li>Grad Accumulation: 8 &rarr; eff.batch=64</li><li>LR: Linear warmup (200 steps) &rarr; constant</li><li>AMP: mixed precision (float16)</li></ul>"
box_p1 = block(0.4, 18.8, 5.8, 2.5, f"<div style='text-align:left;'><b style='color:#E65100;font-size:14px;margin-left:15px;'>Phase 1: FM Frozen</b><br>{p1_txt}</div>", fill="#FFF9C4", stroke="#F9A825", font_color="#424242", align="left")

p2_txt = "<ul><li>All parameters unfrozen (~173M)</li><li>Epoch-based: max 50 epochs</li><li>Early stopping: patience=10, delta=1e-4</li><li>Differential LR: ViT (2e-5), REPA+Cls (2e-4)</li><li>Grad Accumulation: 4 &rarr; eff.batch=32</li></ul>"
box_p2 = block(6.5, 18.8, 5.9, 2.5, f"<div style='text-align:left;'><b style='color:#1B5E20;font-size:14px;margin-left:15px;'>Phase 2: Full Unfreezing</b><br>{p2_txt}</div>", fill="#C8E6C9", stroke="#2E7D32", font_color="#424242", align="left")

arrow(box_p1, box_p2, color="#F57F17", lw=3)

loss_txt = """<div style='text-align:center;'><b>Loss Function</b><br><br>
<div style='background-color:#BF360C;color:white;padding:5px;border-radius:5px;margin-bottom:5px;'><b>AsymmetricBCELoss</b><br>L = -w_pos * (1-p)<sup>&gamma;+</sup> y log(p) - p<sup>&gamma;-</sup> (1-y) log(1-p)</div>
<div style='background-color:#D84315;color:white;padding:5px;border-radius:5px;margin-bottom:5px;'><b>Cosine Similarity Alignment Loss</b><br>L_align = 1 - cos(aligned, f_sig.detach())</div>
<div style='background-color:#E64A19;color:white;padding:5px;border-radius:5px;'><b>Combined: L_total = L_bce + &lambda; * L_align</b></div></div>"""
block(0.4, 21.5, 5.8, 2.3, loss_txt, fill="#FFCCBC", stroke="#BF360C", font_color="#BF360C")

eval_txt = """<div style='text-align:left;padding:10px;'><b style='color:#0277BD;font-size:14px;'>Evaluation &amp; Metrics</b><hr/>
<ul><li><b>Primary:</b> AUROC (best checkpoint)</li><li><b>Official:</b> TPR@5% (PhysioNet 2025)</li></ul>
<b>Threshold Strategies:</b><br>1. Youden: argmax(TPR - FPR)<br>2. Min Recall &ge; 0.99 &rarr; max precision<br>3. Min Precision &ge; 0.30 &rarr; max recall<br>4. Max F1 (scan all)<br>5. Fixed T = 0.5</div>"""
block(6.5, 21.5, 5.9, 2.3, eval_txt, fill="#E1F5FE", stroke="#0277BD", font_color="#0277BD", align="left")


xml_content = builder.build()
with open(r'D:\IIT\L6\FYP\ChagaSight\checkpoints_new\full_p2_50epochs\plots\final_v2\chagasight_architecture_diagram.drawio', 'w', encoding='utf-8') as f:
    f.write(xml_content)
print("Saved drawio file successfully.")

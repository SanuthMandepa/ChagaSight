import base64

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
        
        value_esc = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
        
        cell = f'<mxCell id="{cell_id}" value="{value_esc}" style="{style}" parent="{parent}"{vertex_attr}{edge_attr}{src_attr}{tgt_attr}>{geom}</mxCell>'
        self.cells.append(cell)
        return cell_id

    def build(self):
        header = """<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="Electron" modified="2023-10-01T00:00:00.000Z" agent="Mozilla/5.0" version="21.6.8" type="device">
  <diagram id="diagram-id" name="ChagaSight">
    <mxGraphModel dx="1400" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1500" pageHeight="1000" math="0" shadow="0">
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

def rect(x, y, w, h, text, fill, stroke, font_color="#000", fs=12, align="center", val_align="middle", dashed=False, bold=False):
    style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontColor={font_color};fontSize={fs};align={align};verticalAlign={val_align};strokeWidth=1.5;"
    if dashed: style += "dashed=1;"
    if bold: style += "fontStyle=1;"
    return builder.add_cell(text, x, y, w, h, style)

def label(x, y, w, h, text, font_color="#000", fs=16, bold=True, align="left"):
    style = f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor={font_color};fontSize={fs};"
    if bold: style += "fontStyle=1;"
    return builder.add_cell(text, x, y, w, h, style)

def arrow(src, tgt, waypoints=None, color="#000000", lw=2):
    style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};strokeWidth={lw};endArrow=block;endFill=1;"
    return builder.add_cell("", 0, 0, 0, 0, style, edge="1", source=src, target=tgt, waypoints=waypoints)

# Colors matching friend's aesthetic
BL_T = "#2874A6"
BL_B = "#5DADE2"
BL_F = "#EBF5FB"
BL_F2= "#D6EAF8"

GR_T = "#1E8449"
GR_B = "#58D68D"
GR_F = "#EAFAF1"
GR_F2= "#D5F5E3"

PU_T = "#6C3483"
PU_B = "#AF7AC5"
PU_F = "#F5EEF8"

RE_T = "#B03A2E"
RE_B = "#F1948A"
RE_F = "#FDEDEC"

# Vectors Base64
vec_b = base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 100"><rect x="0" y="0" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="0" y="20" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="0" y="40" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="0" y="60" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="0" y="80" width="20" height="20" fill="#85C1E9" stroke="#333"/></svg>').decode()
vec_g = base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 100"><rect x="0" y="0" width="20" height="20" fill="#82E0AA" stroke="#333"/><rect x="0" y="20" width="20" height="20" fill="#82E0AA" stroke="#333"/><rect x="0" y="40" width="20" height="20" fill="#82E0AA" stroke="#333"/><rect x="0" y="60" width="20" height="20" fill="#82E0AA" stroke="#333"/><rect x="0" y="80" width="20" height="20" fill="#82E0AA" stroke="#333"/></svg>').decode()
vec_c = base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 20"><rect x="0" y="0" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="20" y="0" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="40" y="0" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="60" y="0" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="80" y="0" width="20" height="20" fill="#85C1E9" stroke="#333"/><rect x="100" y="0" width="20" height="20" fill="#82E0AA" stroke="#333"/><rect x="120" y="0" width="20" height="20" fill="#82E0AA" stroke="#333"/><rect x="140" y="0" width="20" height="20" fill="#82E0AA" stroke="#333"/><rect x="160" y="0" width="20" height="20" fill="#82E0AA" stroke="#333"/><rect x="180" y="0" width="20" height="20" fill="#82E0AA" stroke="#333"/></svg>').decode()

sig_svg = base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20" fill="none" stroke="#1E8449" stroke-width="1.5"><path d="M0,10 L10,10 L15,5 L20,15 L25,0 L30,20 L35,10 L100,10"/></svg>').decode()

grid_svg = base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><rect width="40" height="40" fill="#fff" stroke="#ccc"/><line x1="10" y1="0" x2="10" y2="40" stroke="#eee"/><line x1="20" y1="0" x2="20" y2="40" stroke="#eee"/><line x1="30" y1="0" x2="30" y2="40" stroke="#eee"/><line x1="0" y1="10" x2="40" y2="10" stroke="#eee"/><line x1="0" y1="20" x2="40" y2="20" stroke="#eee"/><line x1="0" y1="30" x2="40" y2="30" stroke="#eee"/><path d="M0,20 L10,20 L15,10 L20,30 L25,20 L40,20" fill="none" stroke="#E74C3C" stroke-width="1.5"/></svg>').decode()

# ====================================================================================
# A. Image Branch
# ====================================================================================
label(30, 20, 300, 40, "A. Image Branch<br>(2D-ViT Contour Image)", font_color=BL_T, fs=18)
box_a = rect(30, 70, 830, 310, "", "none", BL_B)

in_a = rect(50, 90, 150, 270, "<b>Input: 12-lead WCT Image</b><br><span style='font-size:11px;'>(B, 3, 24, 2048)</span><hr><br><img src='data:image/svg+xml;base64,"+grid_svg+"' width='120' height='40'/><br><span style='font-size:10px;'>RA-referenced view</span><br><br><img src='data:image/svg+xml;base64,"+grid_svg+"' width='120' height='40'/><br><span style='font-size:10px;'>LA-referenced view</span><br><br><img src='data:image/svg+xml;base64,"+grid_svg+"' width='120' height='40'/><br><span style='font-size:10px;'>LL-referenced view</span>", "none", "#555", val_align="top")
bb_a = rect(230, 90, 520, 270, "<div style='margin-top:5px;font-weight:bold;color:"+BL_T+";font-size:16px;'>2D-ViT Backbone (per channel)</div>", "none", BL_B, val_align="top")

pa_embed = rect(250, 140, 100, 190, "<b>Patch<br>Embedding<br>(8x64)</b><hr><br><br><span style='font-size:11px;'>3 x 32 = 96<br>patches</span>", "none", "#555")
label(240, 340, 120, 20, "Embed Dim = 768", fs=12, align="center")

pa_pos = rect(370, 170, 50, 130, "+ Pos<br>Embed", BL_F, BL_B, BL_T)

tx_a = rect(440, 130, 160, 210, "<div style='margin-top:5px;font-weight:bold;color:"+BL_T+";font-size:14px;'>Transformer Encoder (12 layers)</div>", BL_F, BL_B, val_align="top")
mhsa_a = rect(455, 170, 130, 30, "Multi-Head Self-Attention", "#fff", BL_B, BL_T, fs=10)
an1_a  = rect(455, 215, 130, 25, "Add & Norm", BL_F2, BL_B, BL_T, fs=10)
mlp_a  = rect(455, 255, 130, 30, "MLP (768 &rarr; 3072)", "#fff", BL_B, BL_T, fs=10)
an2_a  = rect(455, 300, 130, 25, "Add & Norm", BL_F2, BL_B, BL_T, fs=10)
arrow(mhsa_a, an1_a)
arrow(an1_a, mlp_a)
arrow(mlp_a, an2_a)
label(585, 135, 30, 20, "<b>x 12</b>", fs=12)

aol_a = rect(640, 180, 80, 110, "Aggregation<br>of Layers<br>(AoL)<hr><span style='font-size:10px;'>Mean Pooling<br>stack 12<br>layers</span>", "none", "#555")

arrow(in_a, pa_embed)
arrow(pa_embed, pa_pos)
arrow(pa_pos, tx_a)
arrow(tx_a, aol_a)

vec_a_img = f"<img src='data:image/svg+xml;base64,{vec_b}' width='20' height='100'/>"
out_vec_a = builder.add_cell(vec_a_img, 770, 185, 20, 100, "text;html=1;")
arrow(aol_a, out_vec_a)
label(735, 290, 100, 40, "<b>Image Feature</b><br>f<sub>img</sub><br>(B, 768)", align="center", fs=12)

conf_a = rect(230, 390, 630, 30, "<b>2D-ViT Configuration:</b>  EMBED_DIM = 768  |  DEPTHS = 12  |  NUM_HEADS = 12  |  PATCH = (8, 64)", "none", BL_B, BL_T, dashed=True)


# ====================================================================================
# B. Signal Branch
# ====================================================================================
label(30, 450, 300, 40, "B. Signal Branch<br>(1D-ViT FM Pathway)", font_color=GR_T, fs=18)
box_b = rect(30, 500, 830, 310, "", "none", GR_B)

in_b = rect(50, 520, 150, 270, "<b>Input: 12-lead ECG Signal</b><br><span style='font-size:11px;'>(B, 12, 1000)</span><hr><br><br>Lead I  <img src='data:image/svg+xml;base64,"+sig_svg+"' width='80' height='15'/><br><br>Lead II  <img src='data:image/svg+xml;base64,"+sig_svg+"' width='80' height='15'/><br><br>&#8942;<br><br>Lead V6 <img src='data:image/svg+xml;base64,"+sig_svg+"' width='80' height='15'/><br><br><span style='font-size:10px;'>Time (10s)</span>", "none", "#555", val_align="top")
bb_b = rect(230, 520, 520, 270, "<div style='margin-top:5px;font-weight:bold;color:"+GR_T+";font-size:16px;'>1D-ViT FM Backbone (per lead)</div>", "none", GR_B, val_align="top")

pa_embed_b = rect(250, 570, 100, 190, "<b>PatchEmbed1D<br>(k=50, s=50)</b><hr><br><br><span style='font-size:11px;'>12 x 20 = 240<br>patches</span>", "none", "#555")
label(240, 770, 120, 20, "Embed Dim = 768", fs=12, align="center")

pa_pos_b = rect(370, 600, 50, 130, "+ Pos<br>Embed<hr>+ Lead<br>Embed", GR_F, GR_B, GR_T)

tx_b = rect(440, 560, 160, 210, "<div style='margin-top:5px;font-weight:bold;color:"+GR_T+";font-size:14px;'>Transformer Encoder (12 layers)</div>", GR_F, GR_B, val_align="top")
mhsa_b = rect(455, 600, 130, 30, "Multi-Head Self-Attention", "#fff", GR_B, GR_T, fs=10)
an1_b  = rect(455, 645, 130, 25, "Add & Norm", GR_F2, GR_B, GR_T, fs=10)
mlp_b  = rect(455, 685, 130, 30, "MLP (768 &rarr; 3072)", "#fff", GR_B, GR_T, fs=10)
an2_b  = rect(455, 730, 130, 25, "Add & Norm", GR_F2, GR_B, GR_T, fs=10)
arrow(mhsa_b, an1_b)
arrow(an1_b, mlp_b)
arrow(mlp_b, an2_b)
label(585, 565, 30, 20, "<b>x 12</b>", fs=12)

aol_b = rect(640, 610, 80, 110, "Aggregation<br>of Layers<br>(AoL)<hr><span style='font-size:10px;'>Mean Pooling<br>stack 12<br>layers</span>", "none", "#555")

arrow(in_b, pa_embed_b)
arrow(pa_embed_b, pa_pos_b)
arrow(pa_pos_b, tx_b)
arrow(tx_b, aol_b)

vec_b_img = f"<img src='data:image/svg+xml;base64,{vec_g}' width='20' height='100'/>"
out_vec_b = builder.add_cell(vec_b_img, 770, 615, 20, 100, "text;html=1;")
arrow(aol_b, out_vec_b)
label(735, 720, 100, 40, "<b>Signal Feature</b><br>f<sub>sig</sub><br>(B, 768)", align="center", fs=12)

conf_b = rect(230, 820, 630, 30, "<b>1D-ViT FM Configuration:</b>  EMBED_DIM = 768  |  DEPTHS = 12  |  NUM_HEADS = 12  |  PATCH = 50  |  LEADS = 12", "none", GR_B, GR_T, dashed=True)


# ====================================================================================
# C. Alignment & Fusion Module
# ====================================================================================
label(900, 250, 250, 40, "C. REPA Alignment &amp;<br>Fusion Module", font_color=PU_T, fs=18)
box_c = rect(900, 310, 260, 460, "", "none", PU_B)

repa_box = rect(920, 330, 220, 120, "", PU_F, PU_B)
label(920, 330, 220, 30, "REPA Alignment Module", font_color=PU_T, align="center", fs=14)
repa_conv = rect(940, 360, 180, 25, "DepthwiseConv1d (768&rarr;768)", "#fff", PU_B, PU_T, fs=11)
repa_silu = rect(940, 395, 180, 25, "SiLU Activation", "#fff", PU_B, PU_T, fs=11)
repa_lin  = rect(940, 430, 180, 25, "Linear (768&rarr;768)", "#fff", PU_B, PU_T, fs=11)
arrow(repa_conv, repa_silu)
arrow(repa_silu, repa_lin)

arrow(out_vec_a, repa_conv, waypoints=[(880, 235), (880, 372)])

demo_box = rect(920, 600, 220, 150, "", PU_F, PU_B)
label(920, 600, 220, 30, "Demographics (FiLM)", font_color=PU_T, align="center", fs=14)
demo_in  = rect(940, 630, 60, 40, "Input:<br>Age, Sex", "#fff", PU_B, PU_T, fs=11)
demo_mlp = rect(1020, 630, 100, 40, "MLP &rarr; &gamma;, &beta;", "#fff", PU_B, PU_T, fs=11)
arrow(demo_in, demo_mlp)
demo_film = rect(940, 690, 180, 40, "Modality Weighting<br>f<sub>sig_mod</sub> = &gamma; &odot; f<sub>sig</sub> + &beta;", "#fff", PU_B, PU_T, fs=12)
arrow(demo_mlp, demo_film)

arrow(out_vec_b, demo_film, waypoints=[(880, 665), (880, 710)])

concat_box = rect(920, 480, 220, 100, "", PU_F, PU_B)
label(920, 480, 220, 30, "Concatenation", font_color=PU_T, align="center", fs=14)
vec_c_img = f"<img src='data:image/svg+xml;base64,{vec_c}' width='200' height='20'/>"
out_vec_c = builder.add_cell(vec_c_img, 930, 520, 200, 20, "text;html=1;")
label(930, 550, 200, 20, "Fused Feature f<sub>fused</sub> (B, 1536)", font_color=PU_T, align="center", fs=12)

arrow(repa_box, concat_box)
arrow(demo_box, concat_box)


# ====================================================================================
# D. Classification Head
# ====================================================================================
label(1220, 310, 200, 40, "D. Classification Head", font_color=RE_T, fs=18)
box_d = rect(1220, 360, 240, 390, "", "none", RE_B)

l1 = rect(1250, 380, 180, 35, "Linear (1536 &rarr; 512)", RE_F, RE_B, RE_T)
a1 = rect(1250, 430, 180, 30, "ReLU", RE_F, RE_B, RE_T)
d1 = rect(1250, 475, 180, 30, "Dropout (p = 0.3)", RE_F, RE_B, RE_T)
l2 = rect(1250, 520, 180, 35, "Linear (512 &rarr; 256)", RE_F, RE_B, RE_T)
a2 = rect(1250, 570, 180, 30, "ReLU", RE_F, RE_B, RE_T)
d2 = rect(1250, 615, 180, 30, "Dropout (p = 0.3)", RE_F, RE_B, RE_T)
l3 = rect(1250, 660, 180, 35, "Linear (256 &rarr; 1)", RE_F, RE_B, RE_T)

arrow(l1, a1, color=RE_T)
arrow(a1, d1, color=RE_T)
arrow(d1, l2, color=RE_T)
arrow(l2, a2, color=RE_T)
arrow(a2, d2, color=RE_T)
arrow(d2, l3, color=RE_T)

arrow(concat_box, l1, waypoints=[(1180, 530), (1180, 397)])

logit = builder.add_cell("<b>Logits<br>(B, 1)</b><br><br><img src='data:image/svg+xml;base64,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 40 20\"><rect x=\"0\" y=\"0\" width=\"40\" height=\"20\" rx=\"5\" ry=\"5\" fill=\"#fff\" stroke=\"#333\"/><circle cx=\"20\" cy=\"10\" r=\"6\" fill=\"#E74C3C\"/></svg>' width='40' height='20'/>", 1290, 760, 100, 60, "text;html=1;align=center;")
arrow(l3, logit)
sig = label(1290, 830, 100, 20, "Sigmoid", fs=14, align="center")
builder.add_cell("", 0,0,0,0, "edgeStyle=orthogonalEdgeStyle;endArrow=block;html=1;strokeColor=#000;", edge="1", source=logit, target=sig)

pred = label(1270, 870, 140, 40, "<b>Prediction</b><br>(Chagas Pos/Neg)", fs=14, align="center")
builder.add_cell("", 0,0,0,0, "edgeStyle=orthogonalEdgeStyle;endArrow=block;html=1;strokeColor=#000;", edge="1", source=sig, target=pred)


# ====================================================================================
# Legend & Overall Input Output
# ====================================================================================
leg_box = rect(30, 880, 830, 90, "", "none", "#555")
rect(50, 910, 20, 20, "", "none", BL_B)
label(80, 910, 180, 20, "Image branch<br>(2D-ViT)", fs=12, bold=False)

rect(260, 910, 20, 20, "", "none", GR_B)
label(290, 910, 180, 20, "Signal branch<br>(1D-ViT FM)", fs=12, bold=False)

rect(480, 910, 20, 20, "", "none", PU_B)
label(510, 910, 180, 20, "Fusion module<br>(REPA + Demographics)", fs=12, bold=False)

rect(700, 910, 20, 20, "", "none", RE_B)
label(730, 910, 100, 20, "Classification head", fs=12, bold=False)

io_box = rect(900, 880, 560, 90, "", "#F8F9F9", "#555")
label(900, 890, 560, 20, "Overall Input and Output", fs=14, bold=True, align="center")

label(920, 920, 60, 20, "Inputs:", fs=12, bold=False)
label(950, 940, 200, 20, "&bull; Images: (B, 3, 24, 2048)<br>&bull; Signal: (B, 12, 1000)<br>&bull; Demo: Age, Sex (B,)", fs=12, bold=False)

builder.add_cell("", 1200, 920, 0, 40, "edgeStyle=none;endArrow=none;html=1;strokeColor=#333;", edge="1", waypoints=[(1200, 920), (1200, 960)])

label(1220, 920, 60, 20, "Output:", fs=12, bold=False)
label(1250, 940, 200, 20, "Probability of<br>Chagas disease", fs=12, bold=False)


xml_content = builder.build()
with open(r'D:\IIT\L6\FYP\ChagaSight\checkpoints_new\full_p2_50epochs\plots\final_v2\chagasight_architecture_v3.drawio', 'w', encoding='utf-8') as f:
    f.write(xml_content)
print("Saved drawio file successfully.")

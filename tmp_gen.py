import xml.etree.ElementTree as ET

mxfile = ET.Element("mxfile", host="Electron", agent="Mozilla/5.0")
diagram = ET.SubElement(mxfile, "diagram", name="Page-1", id="chagasight-dl")
mxGraphModel = ET.SubElement(diagram, "mxGraphModel", dx="1000", dy="1000", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1400", pageHeight="827", math="0", shadow="0")
root = ET.SubElement(mxGraphModel, "root")
ET.SubElement(root, "mxCell", id="0")
ET.SubElement(root, "mxCell", id="1", parent="0")

def add_node(id, value, x, y, w, h, style, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=id, value=value, style=style, vertex="1", parent=parent)
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})

def add_edge(id, source, target, style, parent="1"):
    cell = ET.SubElement(root, "mxCell", id=id, style=style, edge="1", source=source, target=target, parent=parent)
    ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})

nodes = [
    ("n_12lead", "12 lead ECG", 20, 260, 80, 40, "ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;"),
    ("b_prep", "Preprocessing", 150, 100, 220, 520, "rounded=1;whiteSpace=wrap;html=1;verticalAlign=bottom;fillColor=#fce4ec;strokeColor=#c2185b;"),
    ("b_dl", "Deep Learning Model", 400, 100, 950, 520, "rounded=1;whiteSpace=wrap;html=1;verticalAlign=bottom;fillColor=none;strokeColor=#000000;"),
    
    ("n_filter", "Filter", 220, 130, 80, 30, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_500hz", "500 Hz\nresample", 170, 200, 80, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_100hz", "100 Hz\nresample", 270, 200, 80, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_10s", "10s\nsampling/ padding", 200, 280, 120, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_3s", "3s\nsampling", 170, 360, 80, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_norm", "Normalization", 270, 360, 100, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_mixup", "Lead-\nmixup", 170, 440, 80, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_img_construct", "Image-\nconstruction", 160, 540, 100, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    
    ("n_fm_enc", "FM Encoder (ECG\nFoundational Model)\n<font color=\"#0000FF\"><span style=\"font-size: 10px\">Pretrained: ST-MEM | AoL aggregation</span></font>", 430, 380, 200, 50, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_x_orig", "X\n(Original Feature\nVector)", 650, 380, 120, 50, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_x_mod", "X'\n(Modified Feature\nVector)", 850, 380, 120, 50, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    
    ("dummy_x", "", 800, 405, 1, 1, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    
    ("n_demo_enc", "Demographic\nEncoder\n(Age/ Sex)", 750, 180, 100, 50, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_gamma", "γ (Gamma)", 700, 260, 80, 30, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_beta", "β (Beta)", 800, 260, 80, 30, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    
    ("n_cnn", "CNN\nStem", 480, 540, 80, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_vit", "ViT (Vision Transformer)\n<font color=\"#0000FF\"><span style=\"font-size: 10px\">Pretrained: MAE | AoL aggregation</span></font>", 600, 535, 200, 50, "rounded=0;whiteSpace=wrap;html=1;fillColor=#e1bee7;strokeColor=#9c27b0;"),
    ("n_vis_feat", "Vision Features", 850, 540, 100, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    
    ("n_proj_head", "Projection\nHead\n(REPA)", 1000, 460, 100, 50, "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffff8d;strokeColor=#ffeb3b;dashed=1;"),
    ("n_class_head", "Classification\nhead", 1150, 385, 100, 40, "rounded=0;whiteSpace=wrap;html=1;fillColor=#c5cae9;strokeColor=#3f51b5;"),
    
    ("n_prob", "Chagas\nprobability", 1150, 250, 100, 40, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_cos_sim", "Cosine Similarity", 1000, 250, 120, 30, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    
    ("n_loss_box", "", 820, 120, 470, 40, "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;"),
    ("n_loss_txt", "Loss =", 840, 125, 60, 30, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_proj_loss_txt", "Projection Loss", 980, 125, 120, 30, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;"),
    ("n_bce_loss_txt", "+ Asymmetric BCE loss", 1120, 125, 150, 30, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;")
]

def make_edge(name, src, tgt, exit='', entry='', extra=''):
    st = f"endArrow=classic;html=1;rounded=0;{exit}{entry}{extra}"
    return (name, src, tgt, st)

def make_orth(name, src, tgt, exit='', entry=''):
    st = f"edgeStyle=orthogonalEdgeStyle;endArrow=classic;html=1;rounded=0;orthogonalLoop=1;jettySize=auto;{exit}{entry}"
    return (name, src, tgt, st)

edges = [
    make_orth("e_in_filt", "n_12lead", "n_filter", "exitX=1;exitY=0.5;dx=0;dy=0;", "entryX=0;entryY=0.5;dx=0;dy=0;"),
    make_orth("e_filt_500", "n_filter", "n_500hz", "exitX=0.25;exitY=1;dx=0;dy=0;", "entryX=0.5;entryY=0;dx=0;dy=0;"),
    make_orth("e_filt_100", "n_filter", "n_100hz", "exitX=0.75;exitY=1;dx=0;dy=0;", "entryX=0.5;entryY=0;dx=0;dy=0;"),
    make_orth("e_500_10s", "n_500hz", "n_10s", "exitX=0.5;exitY=1;dx=0;dy=0;", "entryX=0.25;entryY=0;dx=0;dy=0;"),
    make_orth("e_100_10s", "n_100hz", "n_10s", "exitX=0.5;exitY=1;dx=0;dy=0;", "entryX=0.75;entryY=0;dx=0;dy=0;"),
    
    make_orth("e_10s_3s", "n_10s", "n_3s", "exitX=0.25;exitY=1;dx=0;dy=0;", "entryX=0.5;entryY=0;dx=0;dy=0;"),
    make_orth("e_10s_norm", "n_10s", "n_norm", "exitX=0.75;exitY=1;dx=0;dy=0;", "entryX=0.5;entryY=0;dx=0;dy=0;"),
    
    make_edge("e_3s_mixup", "n_3s", "n_mixup"),
    make_edge("e_mixup_img", "n_mixup", "n_img_construct"),
    
    make_edge("e_norm_fm", "n_norm", "n_fm_enc"),
    make_edge("e_fm_xorig", "n_fm_enc", "n_x_orig"),
    make_edge("e_xorig_dummy", "n_x_orig", "dummy_x"),
    make_edge("e_dummy_xmod", "dummy_x", "n_x_mod"),
    
    make_orth("e_demo_gamma", "n_demo_enc", "n_gamma", "exitX=0.25;exitY=1;dx=0;dy=0;", "entryX=0.5;entryY=0;dx=0;dy=0;"),
    make_orth("e_demo_beta", "n_demo_enc", "n_beta", "exitX=0.75;exitY=1;dx=0;dy=0;", "entryX=0.5;entryY=0;dx=0;dy=0;"),
    make_orth("e_gamma_cross", "n_gamma", "dummy_x", "exitX=0.5;exitY=1;dx=0;dy=0;", "entryX=0;entryY=0.5;dx=0;dy=0;"),
    make_orth("e_beta_cross", "n_beta", "dummy_x", "exitX=0.5;exitY=1;dx=0;dy=0;", "entryX=0;entryY=0.5;dx=0;dy=0;"),
    
    make_edge("e_img_cnn", "n_img_construct", "n_cnn"),
    make_edge("e_cnn_vit", "n_cnn", "n_vit"),
    make_edge("e_vit_vis", "n_vit", "n_vis_feat"),
    
    make_edge("e_vis_proj", "n_vis_feat", "n_proj_head"),
    
    make_edge("e_proj_cos", "n_proj_head", "n_cos_sim"),
    make_edge("e_xmod_cos", "n_x_mod", "n_cos_sim"),
    
    make_orth("e_vis_class", "n_vis_feat", "n_class_head", "exitX=1;exitY=0.5;dx=0;dy=0;", "entryX=0.5;entryY=1;dx=0;dy=0;"),
    make_edge("e_xmod_class", "n_x_mod", "n_class_head"),
    
    make_edge("e_class_prob", "n_class_head", "n_prob"),
    
    make_edge("e_cos_loss", "n_cos_sim", "n_proj_loss_txt"),
    make_edge("e_prob_loss", "n_prob", "n_bce_loss_txt")
]

for n in nodes:
    add_node(*n)

for e in edges:
    add_edge(*e)

xml_str = ET.tostring(mxfile, encoding="unicode", method="xml")

with open(r"D:\IIT\L6\FYP\Draw.io files\ml_corrected_editable.drawio", "w", encoding="utf-8") as f:
    f.write(xml_str)

print("Generated draw.io successfully!")

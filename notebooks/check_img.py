from PIL import Image
img = Image.open(r'D:\IIT\L6\FYP\ChagaSight\checkpoints_new\full_p2_50epochs\plots\final_v2\chagasight_architecture_diagram.png')
print(f'Size: {img.size}')
print(f'DPI: {img.info.get("dpi", "N/A")}')

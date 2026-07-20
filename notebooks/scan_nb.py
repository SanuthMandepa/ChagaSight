import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open(r'D:\IIT\L6\FYP\ChagaSight\notebooks\11_train_final_split.ipynb', encoding='utf-8'))
# Print full content of cell 4 (config)
src = ''.join(nb['cells'][4]['source'])
print(src)

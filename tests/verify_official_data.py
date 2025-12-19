import os
import wfdb

datasets = ['ptbxl', 'sami_trop', 'code15']
base_path = 'data/official_wfdb'

for ds in datasets:
    path = os.path.join(base_path, ds)
    if not os.path.exists(path):
        print(f"❌ {ds.upper()} folder missing")
        continue
    
    # Count .hea files recursively
    hea_count = 0
    sample_path = None
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.hea'):
                hea_count += 1
                if sample_path is None:
                    sample_path = os.path.join(root, f.replace('.hea', ''))
    
    print(f"✅ {ds.upper()}: {hea_count} records created")
    
    if sample_path and hea_count > 0:
        try:
            header = wfdb.rdheader(sample_path)
            chagas_line = [line for line in header.comments if 'Chagas label' in line]
            print(f"   Sample Chagas label: {chagas_line[0] if chagas_line else 'Not found'}")
        except Exception as e:
            print(f"   Error reading header: {e}")
    print("")
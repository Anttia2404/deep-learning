import json
from pathlib import Path

notebook_dir = Path(r'd:\ffpp\cuoi_ki_DL\notebooks')
notebook_paths = list(notebook_dir.glob('*.ipynb'))

cell_3_old = [
    "# ============================================================\n",
    "# SỬA 2 GIÁ TRỊ NÀY CHO ĐÚNG VỚI KAGGLE DATASET CỦA BẠN\n",
    "# ============================================================\n",
    "FFPP_SLUG = 'ffpp-c23-full'       # slug của dataset FF++ data\n",
    "CODE_SLUG  = 'ffpp-training-code' # slug của dataset code repo\n",
    "# ============================================================\n",
    "\n",
    "FFPP_ROOT = f'/kaggle/input/{FFPP_SLUG}'\n",
    "CODE_ROOT = f'/kaggle/input/{CODE_SLUG}'\n",
    "WORK_DIR  = '/kaggle/working/ffpp'\n",
    "\n",
    "print(f'FF++ data  : {FFPP_ROOT}')\n",
    "print(f'Code input : {CODE_ROOT}')\n",
    "print(f'Work dir   : {WORK_DIR}')"
]

cell_3_new = [
    "# ============================================================\n",
    "# SỬA DƯỚI ĐÂY NẾU KAGGLE ĐƯỜNG DẪN KHÁC\n",
    "# ============================================================\n",
    "BASE_INPUT = '/kaggle/input/datasets/votantainu'\n",
    "FFPP_SLUG = 'ffpp-c23-full'       # slug của dataset FF++ data\n",
    "CODE_SLUG  = 'ffpp-training-code' # slug của dataset code repo\n",
    "# ============================================================\n",
    "\n",
    "FFPP_ROOT = f'{BASE_INPUT}/{FFPP_SLUG}'\n",
    "CODE_ROOT = f'{BASE_INPUT}/{CODE_SLUG}'\n",
    "WORK_DIR  = '/kaggle/working/ffpp'\n",
    "\n",
    "print(f'FF++ data  : {FFPP_ROOT}')\n",
    "print(f'Code input : {CODE_ROOT}')\n",
    "print(f'Work dir   : {WORK_DIR}')"
]

cell_8_old = [
    "import json, urllib.request\n",
    "from pathlib import Path\n",
    "\n",
    "splits_dir = Path(FFPP_ROOT) / 'splits'\n",
    "splits_dir.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "SPLIT_URLS = {\n",
    "    'train.json': 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json',\n",
    "    'val.json'  : 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json',\n",
    "    'test.json' : 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json',\n",
    "}\n",
    "\n",
    "# splits_dir nằm trong /kaggle/input → read-only → dùng working dir\n",
    "splits_work = Path('/kaggle/working/splits')\n",
    "splits_work.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "for fname, url in SPLIT_URLS.items():\n",
    "    target_input = splits_dir / fname\n",
    "    target_work  = splits_work / fname\n",
    "    \n",
    "    if target_input.exists():\n",
    "        print(f'✅ {fname} exists in dataset')\n",
    "        # Copy sang working để chắc chắn writable\n",
    "        import shutil\n",
    "        shutil.copy(target_input, target_work)\n",
    "    else:\n",
    "        print(f'⬇️  Downloading {fname} from GitHub...')\n",
    "        urllib.request.urlretrieve(url, target_work)\n",
    "        print(f'✅ {fname} downloaded')\n",
    "    \n",
    "    # Verify\n",
    "    with open(target_work) as f:\n",
    "        data = json.load(f)\n",
    "    print(f'   {fname}: {len(data)} entries\\n')\n",
    "\n",
    "print('Splits ready at:', splits_work)"
]

cell_8_new = [
    "import json, urllib.request\n",
    "from pathlib import Path\n",
    "\n",
    "# splits_dir nằm trong /kaggle/input → read-only → dùng working dir trực tiếp\n",
    "splits_work = Path('/kaggle/working/splits')\n",
    "splits_work.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "SPLIT_URLS = {\n",
    "    'train.json': 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json',\n",
    "    'val.json'  : 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json',\n",
    "    'test.json' : 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json',\n",
    "}\n",
    "\n",
    "for fname, url in SPLIT_URLS.items():\n",
    "    target_work  = splits_work / fname\n",
    "    \n",
    "    # Thử tìm trong input trước, nếu không có sẽ tự tải từ GitHub\n",
    "    target_input = Path(FFPP_ROOT) / 'splits' / fname\n",
    "    if target_input.exists():\n",
    "        print(f'✅ {fname} exists in dataset')\n",
    "        import shutil\n",
    "        shutil.copy(target_input, target_work)\n",
    "    else:\n",
    "        print(f'⬇️  Downloading {fname} from GitHub...')\n",
    "        urllib.request.urlretrieve(url, target_work)\n",
    "        print(f'✅ {fname} downloaded')\n",
    "    \n",
    "    # Verify\n",
    "    with open(target_work) as f:\n",
    "        data = json.load(f)\n",
    "    print(f'   {fname}: {len(data)} entries\\n')\n",
    "\n",
    "print('Splits ready at:', splits_work)"
]

for nb_path in notebook_paths:
    print(f'Processing {nb_path.name}...')
    with nb_path.open('r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # Check if it's Cell 3
            if any('FFPP_SLUG = ' in line for line in cell['source']) and any('CODE_SLUG  = ' in line for line in cell['source']) and not any('BASE_INPUT = ' in line for line in cell['source']):
                cell['source'] = cell_3_new
                modified = True
                print('  -> Updated Cell 3')
            # Check if it's Cell 8
            elif any('splits_dir.mkdir(' in line for line in cell['source']):
                cell['source'] = cell_8_new
                modified = True
                print('  -> Updated Cell 8')
                
    if modified:
        with nb_path.open('w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f'  -> Saved {nb_path.name}')
    else:
        print('  -> No changes needed')

print('All notebooks processed!')

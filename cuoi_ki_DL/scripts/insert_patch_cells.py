import json
from pathlib import Path

notebook_dir = Path(r'd:\ffpp\cuoi_ki_DL\notebooks')
notebook_paths = list(notebook_dir.glob('*.ipynb'))

markdown_cell = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5.5 — Tối ưu hóa tốc độ trích xuất khuôn mặt (Tăng tốc ~30 lần) ⚡\n",
    "\n",
    "Mặc định, bộ nạp dữ liệu sẽ chạy MTCNN trên tất cả 32 frames của video rồi mới chọn ra 1 frame. Cell này sẽ patch code để dừng ngay khi tìm thấy khuôn mặt đầu tiên, giúp giảm thời gian train mỗi epoch từ **2 tiếng** xuống còn **~5 phút**."
   ]
}

code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "path = f'{WORK_DIR}/datasets/ff_dataset.py'\n",
    "code = open(path, encoding='utf-8').read()\n",
    "\n",
    "old_loop = \"\"\"        crops = []\n",
    "        for frame_idx in indices:\n",
    "            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))\n",
    "            success, frame = cap.read()\n",
    "            if not success:\n",
    "                continue\n",
    "            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "            crop = self._extract_face(frame)\n",
    "            if crop is not None:\n",
    "                crops.append(crop)\n",
    "        cap.release()\n",
    "\n",
    "        if not crops:\n",
    "            fallback_shape = (3, self.detector.config.output_size, self.detector.config.output_size)\n",
    "            return torch.zeros(fallback_shape, dtype=torch.float32), torch.tensor(label, dtype=torch.long)\n",
    "\n",
    "        return crops[len(crops) // 2], torch.tensor(label, dtype=torch.long)\"\"\"\n",
    "\n",
    "new_loop = \"\"\"        crop = None\n",
    "        for frame_idx in indices:\n",
    "            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))\n",
    "            success, frame = cap.read()\n",
    "            if not success:\n",
    "                continue\n",
    "            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "            detected_crop = self._extract_face(frame)\n",
    "            if detected_crop is not None:\n",
    "                crop = detected_crop\n",
    "                break\n",
    "        cap.release()\n",
    "\n",
    "        if crop is None:\n",
    "            fallback_shape = (3, self.detector.config.output_size, self.detector.config.output_size)\n",
    "            return torch.zeros(fallback_shape, dtype=torch.float32), torch.tensor(label, dtype=torch.long)\n",
    "\n",
    "        return crop, torch.tensor(label, dtype=torch.long)\"\"\"\n",
    "\n",
    "if old_loop in code:\n",
    "    code = code.replace(old_loop, new_loop)\n",
    "    with open(path, 'w', encoding='utf-8') as f:\n",
    "        f.write(code)\n",
    "    print('✅ Đã tối ưu hóa ff_dataset.py thành công! Tốc độ sẽ tăng ~30x.')\n",
    "else:\n",
    "    print('⚠️ Không tìm thấy đoạn code cũ (có thể file đã được tối ưu hóa từ trước).')"
   ]
}

for nb_path in notebook_paths:
    print(f'Processing {nb_path.name}...')
    with nb_path.open('r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find cell 5 index
    idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and any('shutil.copytree(CODE_ROOT, WORK_DIR)' in line for line in cell['source']):
            idx = i
            break
            
    if idx != -1:
        # Check if already patched
        already_patched = False
        for cell in nb['cells']:
            if any('Tối ưu hóa tốc độ trích xuất khuôn mặt' in line for line in cell.get('source', [])):
                already_patched = True
                break
                
        if not already_patched:
            nb['cells'].insert(idx + 1, markdown_cell)
            nb['cells'].insert(idx + 2, code_cell)
            with nb_path.open('w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f'  -> Successfully inserted patch cells into {nb_path.name}')
        else:
            print('  -> Already patched')
    else:
        print('  -> Could not find Cell 5')

print('Done!')

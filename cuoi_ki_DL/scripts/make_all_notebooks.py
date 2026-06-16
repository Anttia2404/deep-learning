import json

with open(r'd:\ffpp\cuoi_ki_DL\notebooks\kaggle_train_deepfakes.ipynb', encoding='utf-8') as f:
    text = f.read()

# Generate FaceSwap notebook
text_fs = text.replace('Train Deepfakes Detector (Session 1)', 'Train FaceSwap Detector (Session 3)')
text_fs = text_fs.replace('manipulation: Deepfakes', 'manipulation: FaceSwap')
text_fs = text_fs.replace('best_Deepfakes_c23', 'best_FaceSwap_c23')
text_fs = text_fs.replace("'Deepfakes'", "'FaceSwap'")
text_fs = text_fs.replace('"Deepfakes"', '"FaceSwap"')

with open(r'd:\ffpp\cuoi_ki_DL\notebooks\kaggle_train_faceswap.ipynb', 'w', encoding='utf-8') as f:
    f.write(text_fs)

# Generate NeuralTextures notebook
text_nt = text.replace('Train Deepfakes Detector (Session 1)', 'Train NeuralTextures Detector (Session 4)')
text_nt = text_nt.replace('manipulation: Deepfakes', 'manipulation: NeuralTextures')
text_nt = text_nt.replace('best_Deepfakes_c23', 'best_NeuralTextures_c23')
text_nt = text_nt.replace("'Deepfakes'", "'NeuralTextures'")
text_nt = text_nt.replace('"Deepfakes"', '"NeuralTextures"')

with open(r'd:\ffpp\cuoi_ki_DL\notebooks\kaggle_train_neuraltextures.ipynb', 'w', encoding='utf-8') as f:
    f.write(text_nt)

print('Generated FaceSwap and NeuralTextures notebooks locally!')

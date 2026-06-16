import json

with open(r'd:\ffpp\cuoi_ki_DL\notebooks\kaggle_train_deepfakes.ipynb', encoding='utf-8') as f:
    text = f.read()

text = text.replace('Train Deepfakes Detector (Session 1)', 'Train Face2Face Detector (Session 2)')
text = text.replace('manipulation: Deepfakes', 'manipulation: Face2Face')
text = text.replace('best_Deepfakes_c23', 'best_Face2Face_c23')
text = text.replace("'Deepfakes'", "'Face2Face'")
text = text.replace('"Deepfakes"', '"Face2Face"')

with open(r'd:\ffpp\cuoi_ki_DL\notebooks\kaggle_train_face2face.ipynb', 'w', encoding='utf-8') as f:
    f.write(text)

print('Done! kaggle_train_face2face.ipynb created.')

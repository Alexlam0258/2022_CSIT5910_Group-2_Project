from PIL import Image
import os

imgpath = 'C:\\Users\\Lam\\OneDrive - HKUST Connect\\Desktop\\Lecture Note\\CSIT5910\\Project\\data\\jpg'
os.chdir(imgpath)
for _, _, filenames in os.walk(imgpath):
    for file in filenames:
        image = Image.open(file)
        image = image.resize((250, 250))
        image.save(file)

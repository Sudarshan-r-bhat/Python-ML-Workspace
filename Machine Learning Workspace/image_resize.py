from PIL import Image
for i in range(1, 50):
    try:
        image = Image.open(f"C:/Users/Microsoft/Desktop/sudarshan_crop/Sunil_Shetty/ActiOn_{i}.png")
        img = image.resize((100, 100))
        img.save(f"C:/Users/Microsoft/Desktop/sudarshan_crop/Sunil_Shetty/ActiOn_{i}.png")
    except Exception as e:
        continue



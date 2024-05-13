import cv2
import numpy as np

# Wczytanie obrazów
original_image = cv2.imread(r"E:\studia\AAW\2DImages\small_refer_images\anim0.bmp", cv2.IMREAD_GRAYSCALE)
skeleton1 = cv2.imread(r"E:\studia\AAW\2DImages\small_refer_images_skel\anim0.bmp", cv2.IMREAD_GRAYSCALE)
skeleton2 = cv2.imread(r"E:\studia\AAW\2DImages\small_refer_images_res\anim0.bmp", cv2.IMREAD_GRAYSCALE)

# Sprawdzenie, czy obrazy zostały poprawnie wczytane
if original_image is None or skeleton1 is None or skeleton2 is None:
    print("Nie udało się wczytać obrazów.")
    exit()

# Przekształcenie obrazów szkieletyzacji na obrazy 3-kanałowe
skeleton1_color = cv2.cvtColor(skeleton1, cv2.COLOR_GRAY2BGR)
skeleton2_color = cv2.cvtColor(skeleton2, cv2.COLOR_GRAY2BGR)
original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
skeleton1_color[:,:,0] = 0
skeleton1_color[:,:,1] = 0
skeleton2_color[:,:,1] = 0
skeleton2_color[:,:,2] = 0

# Przygotowanie kolorów dla szkieletów
color1 = (0, 255, 0)  # Zielony kolor dla pierwszego szkieletu
color2 = (0, 0, 255)  # Czerwony kolor dla drugiego szkieletu

# Nałożenie szkieletów na oryginalny obraz
overlay1 = cv2.addWeighted(original_image, 0.5, skeleton1_color, 0.5, 0)
overlay2 = cv2.addWeighted(overlay1, 1, skeleton2_color, 0.5, 0)

# Zapisanie obrazu do pliku
cv2.imwrite('nałożony_obraz.bmp', overlay2)
pixel_diff = np.sum(np.abs(skeleton1/255 - skeleton2/255))
print(pixel_diff)
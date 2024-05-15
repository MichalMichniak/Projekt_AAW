import cv2
import os

def create_animation(image_folder, output_video_path, fps=30):
    """
    Tworzy animację poklatkową z serii obrazków.

    :param image_folder: Ścieżka do folderu zawierającego obrazki.
    :param output_video_path: Ścieżka do pliku wyjściowego wideo.
    :param fps: Liczba klatek na sekundę dla animacji.
    """
    # Pobierz listę plików w folderze z obrazkami
    images = [img for img in os.listdir(image_folder) if img.endswith((".bmp", ".jpg", ".jpeg"))]
    images.sort()  # Upewnij się, że obrazy są w odpowiedniej kolejności

    # Sprawdź, czy są jakieś obrazy
    if not images:
        print("Brak obrazów w folderze.")
        return

    # Odczytaj pierwszy obraz, aby uzyskać rozmiar klatek
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Utwórz obiekt VideoWriter do zapisu wideo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Kodek do zapisu wideo
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Dodaj wszystkie obrazy do wideo
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Zwolnij obiekt VideoWriter
    video.release()
    print(f"Animacja zapisana jako {output_video_path}")

# Przykładowe użycie funkcji
image_folder = r"E:\studia\AAW\2DImages\anime"
output_video_path = 'animacja.mp4'
fps = 10  # Liczba klatek na sekundę

create_animation(image_folder, output_video_path, fps)
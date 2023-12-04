import face_recognition
import cv2
import numpy as np
import os
import textwrap
import argparse
import sys

def draw_label(image, text, position, font, font_scale, thickness, box_width):
    lines = textwrap.wrap(text, width=int(box_width / (font_scale * 10)))
    y = position[1]
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_width, text_height = text_size[0], text_size[1]
        line_height = text_height + 10
        x_text_start = position[0] - text_width // 2

        cv2.rectangle(image, (x_text_start, y - line_height), (x_text_start + text_width, y), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, line, (x_text_start, y - 5), font, font_scale, (255, 255, 255), thickness)
        y += line_height

def run(known_folder, unknown_folder, results_folder):

    if not os.path.exists(known_folder):
        sys.exit(f"Erro: Pasta '{known_folder}' não encontrada.")
    if not os.path.exists(unknown_folder):
        sys.exit(f"Erro: Pasta '{unknown_folder}' não encontrada.")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Carregar e aprender a reconhecer os rostos conhecidos
    celebrities_names = os.listdir(known_folder)
    images_path = [os.path.join(known_folder, name) for name in celebrities_names]
    load_face_images = [face_recognition.load_image_file(image_path) for image_path in images_path]
    known_face_encodings = [face_recognition.face_encodings(face_image)[0] for face_image in load_face_images]

    known_face_names = [file.split('.')[0] for file in celebrities_names]

    # Iterar sobre as imagens na pasta
    for image_file in os.listdir(unknown_folder):
        # Carregar a imagem
        image_path = os.path.join(unknown_folder, image_file)
        image = face_recognition.load_image_file(image_path)

        # Localizar rostos e encodings na imagem
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            face_width = right - left
            font_scale = face_width / 200
            thickness = 2

            label_x_center = left + face_width // 2
            label_y_bottom = bottom + 20

            draw_label(image, name, (label_x_center, label_y_bottom), cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness, face_width)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{results_folder}/{image_file}', image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconhecimento facial')
    parser.add_argument('known_folder', help='Pasta com fotos de pessoas conhecidas')
    parser.add_argument('unknown_folder', help='Pasta com fotos de pessoas desconhecidas')
    parser.add_argument('--results_folder', default='results', help='Pasta para salvar os resultados (padrão: results)')

    args = parser.parse_args()

    run(args.known_folder, args.unknown_folder, args.results_folder)

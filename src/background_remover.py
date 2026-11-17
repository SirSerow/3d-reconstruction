import os

from rembg import remove, new_session
from PIL import Image

class BackgroundRemover:
    def __init__(self):
        self.session = new_session()

    def remove_background(self, input_path, output_path):
        image = Image.open(input_path)
        output = remove(image, session=self.session)
        output.save(output_path)


    def remove_background_from_folder(self, input_folder, output_folder):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            self.remove_background(input_path, output_path)

    
if __name__ == "__main__":
    folder_path = "C:/Users/oleg.serov/Projects/scripts/3d-reconstruction/.output/cam_1_1"
    output_folder = "output2"
    remover = BackgroundRemover()
    remover.remove_background_from_folder(folder_path, output_folder)

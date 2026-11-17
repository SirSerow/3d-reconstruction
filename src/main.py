import os
import argparse
import logging
import datetime

from tifffile import logger

from image_splitter import ImageSplitter
from background_remover import BackgroundRemover

def main():

    start_time = datetime.datetime.now()

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    # add file handler
    log_folder = ".logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    file_handler = logging.FileHandler(os.path.join(log_folder, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("Script started")

    # Parse arguments
    parser = argparse.ArgumentParser(description="3D Reconstruction Script")
    parser.add_argument("--input", required=True, help="Input image or video folder")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # Create a dedicated subdir in the output folder for this run
    run_output_path = os.path.join(output_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logger.info(f"Run output path: {run_output_path}")
    if not os.path.exists(run_output_path):
        os.makedirs(run_output_path)


    # Create subfolder for split images
    split_output_path = os.path.join(run_output_path, "split")

    if not os.path.exists(split_output_path):
        os.makedirs(split_output_path)

    logger.info(f"Split output path: {split_output_path}")

    splitter = ImageSplitter(split_output_path)

    splitter.split_and_save(input_path)


    # Create subfolder for background removed images
    bg_removed_output_path = os.path.join(run_output_path, "background_removed")

    if not os.path.exists(bg_removed_output_path):
        os.makedirs(bg_removed_output_path)

    logger.info(f"Background removed output path: {bg_removed_output_path}")

    remover = BackgroundRemover()

    # for each subfolder in the split output path, remove the background and save in a similarly named subfolder in the background removed output path
    for subfolder in os.listdir(split_output_path):

        subfolder_path = os.path.join(split_output_path, subfolder)

        if os.path.isdir(subfolder_path):

            output_subfolder_path = os.path.join(bg_removed_output_path, subfolder)

            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)

            logger.info(f"Processing subfolder: {subfolder_path}")

            remover.remove_background_from_folder(subfolder_path, output_subfolder_path)

    finish_time = datetime.datetime.now()
    logger.info(f"Script finished. Duration: {finish_time - start_time}")


if __name__ == "__main__":
    main()
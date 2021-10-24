
import os
from PIL import Image
import xml.etree.ElementTree as ET
import random
from typing import List, Union
from shutil import copyfile

DATA_DIRECTORY = os.path.join(os.getcwd(), 'archive')

def processing_data(directory:str, training_path:str, testing_path:str) -> List:

    """
    1. Extract all breeds from directory
    2. For each breed open the image and the corresponing annotation
    3. Crop the image to select only the dog in the picture and remove other information
    4. Save to cropped_images sub-folder
    5. Create the Training and Validation directories to use the flow_from_direcotry keras method
    """


    breed_list = os.listdir(os.path.join(os.getcwd(), "archive/images/Images"))
    breed_list.remove('.DS_Store')

    if os.path.isdir('archive/croped_images/') is False:
        os.mkdir('archive/croped_images')


    for breed in breed_list:
        try:
            os.mkdir("archive/croped_images/" + breed)
            for file in os.listdir('archive/annotations/Annotation/{}'.format(breed)):
                print("Processing file ... {}".format(file))
                img = Image.open('archive/images/Images/{}/{}.jpg'.format(breed, file))
                tree = ET.parse('archive/annotations/Annotation/{}/{}'.format(breed, file))
                xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
                xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
                ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
                ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
                img = img.crop((xmin, ymin, xmax, ymax))
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img.save('archive/croped_images/' + breed + '/' + file + '.jpg')
        except FileExistsError:
            pass
    
    for path in [training_path, testing_path]:
        try:
            os.mkdir(os.path.join(directory, path))
        except FileExistsError:
            print("Folder already exists")
            pass
    
    print("Initial Directories created or existed")
    return breed_list


def train_test_split(directory: str, breed: str, sample_size:float) -> Union[List, List]:

    jpgs = os.listdir(os.path.join(directory,'croped_images', breed))
    n_samples = round(len(jpgs) * sample_size, 0)
    training_samples = random.sample(jpgs, int(n_samples))
    testing_samples = list(set(jpgs)-set(training_samples))

    print("Train and Test directories created")
    
    return training_samples, testing_samples
        

def move_to_directories(directory, breed, samples, path) -> None:

    breed_name = '_'.join(breed.split("-")[1:])
    try:
        os.mkdir(os.path.join(directory, path, breed_name))
        for sample in samples:
            copyfile(os.path.join(directory,'croped_images', breed, sample), 
                     os.path.join(directory, path, breed_name, sample))
    except FileExistsError:
        pass 
    print("Move images to train and test directories")

    return None


if __name__ == "__main__":

    breeds = processing_data(directory=DATA_DIRECTORY, training_path='Training_path', testing_path='Testing_path')
    try:
        for breed in breeds:
            train, test = train_test_split(DATA_DIRECTORY, breed, 0.85)
            move_to_directories(DATA_DIRECTORY, breed, train, 'Training_path')
            move_to_directories(DATA_DIRECTORY, breed, test, 'Testing_path')
    except FileExistsError or NotADirectoryError as ee:
        print(ee)
        pass
   
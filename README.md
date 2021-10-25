# Computer Vision - Classification
Classify between different Dog Breeds using Convolutional Neural Networks with Keras and Tensorflow

![Screenshot 2021-10-25 at 1 16 09 PM](https://user-images.githubusercontent.com/31068589/138678161-4b49e9d7-fdac-40a7-aa2b-69cca6c6fb00.png)

## Steps to run the code in this repository

1. Download the data from Kaggle here [Kaggle Dog Breeds Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)
2. Clone the repository
3. Move data into the repository
4. Run the below script to create the new images folder and training and validation directories 
```bash
python data_preprocessing.py
```
You should get something like the below
```bash
archive/  
          Training_path/
                          Afghan_hound/
                                        n02088094_1003.jpg
                                        n02088094_13011.jpg
                                        ...
                          Scotch_terrier/
                                        n02097298_1007.jpg
                                        n02097298_15004.jpg
                                        ...
          Testing_path/
                          Afghan_hound/
                                        n02088094_1023.jpg
                                        n02088094_1406.jpg
                                        ...
                          Scotch_terrier/
                                        n02097298_2083.jpg
                                        n02097298_2998.jpg
                                        ...
                          ...
          croped_images/
          annotations/
          images/
```

5. Now you have all the data structured to run the keras script. Run the below
```bash
python keras_script.py
```
6. The keras_script will create the artifacts necessary to run the Jupyter notebook "Result Exploration.ipynb"


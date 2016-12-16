import numpy as np
import os
from keras.preprocessing import image
import time
import pandas
import threading
import pprint
from PIL import Image
import os

classes_lands = ['badlands', 'butte', 'cliff', 'coast', 'corn_field', 'farm', 'field/wild', 'forest_path', 'forest_road', 'field/cultivated', 'snowfield', 'wheat_field', 'iceberg', 'marsh', 'mountain', 'mountain_snowy', 'pasture', 'rock_arch', 'sea_cliff', 'swamp', 'tree_farm', 'valley', 'volcano']


images_path_train = "/imatge/vgarcia/projects/deep_learning/Places/data/vision/torralba/deeplearning/images256"
labels_path_train = "/imatge/vgarcia/projects/deep_learning/Places/trainvalsplit_places205/train_places205.csv"

images_path_val = "/imatge/vgarcia/projects/deep_learning/Places/data/vision/torralba/deeplearning/images256"
labels_path_val = "/imatge/vgarcia/projects/deep_learning/Places/trainvalsplit_places205/val_places205.csv"


def _init_():
    if not os.path.exists('../datasets/places'):
        os.makedirs('../datasets/places')
    if not os.path.exists('../datasets/places/train'):
        os.makedirs('../datasets/places/train')
    if not os.path.exists('../datasets/places/val'):
        os.makedirs('../datasets/places/val')
    if not os.path.exists('../datasets/places/test'):
        os.makedirs('../datasets/places/test')


def _waitJobs_(jobs):
    for j in jobs:
        j.join()

    return []


def load_img(path, target_size=None):
    '''Load an image into PIL format.
    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    
    img = Image.open(path)
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img

class DataLoader(): 
    def __init__(self, images_path, labels_path, maximum_images, img_size, load_classes = []):
        self.cursor = 0
        self.img_size = img_size

        df = pandas.read_csv(labels_path, sep=" ", header=None, index_col=0)
        self.dirname = images_path

        rows = list(df.index.get_values())

        self.rows = []
        self.epoch = 0
        for row in rows:
            save = False
            for class_ in load_classes:
                if class_ in row: 
                    save = True
            if save or len(load_classes)==0: 
                self.rows.append(row)


        np.random.shuffle(self.rows)

        self.total_num = len(self.rows)
        self.maximum_images = maximum_images
        print("Total number of images "+str(self.total_num))
    
    def load_image_worker(self, path, i, images_atomic):
        try:

            #Load  &  Resize
            x_rgb = load_img(path, target_size=(self.img_size, self.img_size)) 

            #Transpose if theano backend
            x_rgb = image.img_to_array(x_rgb, dim_ordering='tf') 

            #We make sure it is RGB
            if x_rgb.shape[2] > 1:

                images_atomic[i] = [x_rgb, path]
            #else:
                #print "This image is not RGB"
        except:
            print(path+" Error loading this image")
            pass

    def load_data(self, verbose = 1, num_threads = 80):
        
        if verbose:
            print("\nLoading "+str(self.maximum_images)+" images")
            print("Loading from position "+str(self.cursor))


        X_rgb = np.zeros((self.maximum_images, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        i = 0
        diff = 0
        errors = 0
        while i < self.maximum_images:
            images_atomic = {}

            #Initialazing pseudo-atomic variable
            for i2 in range(num_threads):
                images_atomic[i2] = None
            jobs = []

            #Running threads
            for th_i in range(num_threads):
                if (self.cursor+th_i)%self.total_num == 0: 
                    np.random.shuffle(self.rows)
                file = self.rows[(self.cursor+th_i)%self.total_num]
                path = self.dirname+"/"+file
                j = threading.Thread(target=self.load_image_worker, args=(path, th_i, images_atomic))
                j.start()
                jobs.append(j)
            jobs = _waitJobs_(jobs)

            #Extracting results from threads
            for key, counter in zip(images_atomic, range(len(images_atomic.keys()))):
                if images_atomic[key] is not None:
                    X_rgb[i,:,:,:] = images_atomic[key][0]
                    i += 1
                    if i%(self.maximum_images/10)==0 and verbose:
                        print("Loading: "+str(100*i/self.maximum_images)+"%")+ "  Correctly Loaded: "+str(i)+" Errored files: "+str(errors)
                    if i >= self.maximum_images: 
                        diff = len(images_atomic.keys()) - counter - 1 
                        break
                else:
                    errors += 1

            if self.cursor + num_threads - diff >= self.total_num:
                self.epoch += 1
            self.cursor = (self.cursor + num_threads - diff)%self.total_num
            


        return X_rgb

    def getLenData(self):
        return self.total_num




def save_worker(image, path):
    Image.fromarray(image).save(path)



def main(train_images = 200000, val_images = 30000, max_memory_images = 300):
    print "Loading labels.."
    _init_()
    LOADER = DataLoader(images_path = images_path_train, 
                            labels_path = labels_path_train,
                            maximum_images = max_memory_images,
                            img_size = 256)  
    LOADER_VAL = DataLoader(images_path = images_path_val, 
                            labels_path = labels_path_val,
                            maximum_images = max_memory_images,
                            img_size = 256)
    print "Labels loaded.."

    print "*****Saving Training******"
    iters = 0
    jobs = []
    for i in range(train_images/max_memory_images):
        train_batch = LOADER.load_data(verbose = 0)
        if LOADER.epoch > 0: break
        for j in range(train_batch.shape[0]):
            j = threading.Thread(target=save_worker, args=(train_batch[j], '../datasets/places/train/'+str(iters)+'.png'))
            j.start()
            jobs.append(j)
            iters += 1
        jobs = _waitJobs_(jobs)    
        print "Iteration "+str(iters)+" from "+str(train_images)+" completed"

    print "*****Saving Validation******"
    iters = 0
    jobs = []
    for i in range(val_images/max_memory_images):
        val_batch = LOADER_VAL.load_data(verbose = 0)
        if LOADER_VAL.epoch > 0: break
        for j in range(val_batch.shape[0]):
            j = threading.Thread(target=save_worker, args=(val_batch[j], '../datasets/places/val/'+str(iters)+'.png'))
            j.start()
            jobs.append(j)
            iters += 1
        jobs = _waitJobs_(jobs)
        print "Iteration "+str(iters)+" from "+str(val_images)+" completed"

    print "Finished"

if __name__ == "__main__":
    main()

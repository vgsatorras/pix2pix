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


def _init_(vocab_labels = None):
    if not os.path.exists('../datasets/places'):
        os.makedirs('../datasets/places')
    if not os.path.exists('../datasets/places/train'):
        os.makedirs('../datasets/places/train')
    if not os.path.exists('../datasets/places/val'):
        os.makedirs('../datasets/places/val')
    if not os.path.exists('../datasets/places/test'):
        os.makedirs('../datasets/places/test')

    if vocab_labels is not None:
        for label in vocab_labels:
            if not os.path.exists('../datasets/places'):
                os.makedirs('../datasets/places')
            if not os.path.exists('../datasets/places/train'+'/'+label):
                os.makedirs('../datasets/places/train'+'/'+label)
            if not os.path.exists('../datasets/places/val'+'/'+label):
                os.makedirs('../datasets/places/val'+'/'+label)
            if not os.path.exists('../datasets/places/test'+'/'+label):
                os.makedirs('../datasets/places/test'+'/'+label)


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

        self.vocab_labels = []
        self.vocab_single = []
        self.extract_labels()

        np.random.shuffle(self.rows)

        self.total_num = len(self.rows)
        self.maximum_images = maximum_images
        print("Total number of images "+str(self.total_num))


    def extract_labels(self):
        rows_aux = self.rows
        self.rows = []
        for i in range(len(rows_aux)):
            label = ''
            file_parsed = rows_aux[i].split('/')
            for j in range(len(file_parsed)-2):
                if len(label)==0:
                    label = file_parsed[j+1]
                else:
                    label = label+'#'+file_parsed[j+1]
                self.vocab_single.append(file_parsed[j+1])

            self.rows.append([rows_aux[i], label])
            self.vocab_labels.append(label)


        self.vocab_single = list(set(self.vocab_single))
        self.vocab_labels = list(set(self.vocab_labels))
    
    def load_image_worker(self, row, i, images_atomic):
        '''try:'''
        [file, label] = row
        path = self.dirname+"/"+file

        #Load  &  Resize
        x_rgb = load_img(path, target_size=(self.img_size, self.img_size)) 

        #Transpose if theano backend
        x_rgb = image.img_to_array(x_rgb, dim_ordering='tf') 

        #We make sure it is RGB
        if x_rgb.shape[2] > 1:

            #rgb2lab & noralization & transpose
            images_atomic[i] = [x_rgb, path, label]
        #else:
            #print "This image is not RGB"
        '''except:
            print(path+" Error loading this image")
            pass'''

    def load_data(self, verbose = 1, num_threads = 80):
        
        if verbose:
            print("\nLoading "+str(self.maximum_images)+" images")
            print("Loading from position "+str(self.cursor))


        X_rgb = np.zeros((self.maximum_images, self.img_size, self.img_size, 3), dtype=np.uint8)
        paths = []
        labels_response = []
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
                row = self.rows[(self.cursor+th_i)%self.total_num]
                j = threading.Thread(target=self.load_image_worker, args=(row, th_i, images_atomic))
                j.start()
                jobs.append(j)
            jobs = _waitJobs_(jobs)

            #Extracting results from threads
            for key, counter in zip(images_atomic, range(len(images_atomic.keys()))):
                if images_atomic[key] is not None:
                    X_rgb[i,:,:,:] = images_atomic[key][0]
                    paths.append(images_atomic[key][1])
                    labels_response.append(images_atomic[key][2])
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
            


        return X_rgb, paths, labels_response

    def getLenData(self):
        return self.total_num




def save_worker(image, root, label, name):
    path = root+label+'/'+name
    Image.fromarray(image).save(path)



def main(train_images = 300000, val_images = 20000, max_memory_images = 300):
    print "Loading labels.."
    LOADER = DataLoader(images_path = images_path_train, 
                            labels_path = labels_path_train,
                            maximum_images = max_memory_images,
                            img_size = 256)  
    LOADER_VAL = DataLoader(images_path = images_path_val, 
                            labels_path = labels_path_val,
                            maximum_images = max_memory_images,
                            img_size = 256)

    print "\nVocab labels"
    print LOADER.vocab_labels
    print "\nVocab single"
    print LOADER.vocab_single
    _init_(LOADER.vocab_labels)

    f = open('../datasets/places/classes.txt', 'w')
    f.write('\n\nvocab_labels\n')
    f.write(', '.join(LOADER.vocab_labels))
    f.write('\n\nvocab_single\n')
    f.write(', '.join(LOADER.vocab_single))
    f.close()
    time.sleep(4)
    print "Labels loaded.."

    f = open('../datasets/places/creation_log.txt', 'w')
    f.write("\n*****Saving Training******\n")
    print "*****Saving Training******"
    iters = 0
    jobs = []
    for i in range(train_images/max_memory_images):
        train_batch, paths, labels = LOADER.load_data(verbose = 0)
        if LOADER.epoch > 0: break
        for j in range(train_batch.shape[0]):
            j = threading.Thread(target=save_worker, args=(train_batch[j], '../datasets/places/train/', labels[j], str(iters)+'.png'))
            j.start()
            jobs.append(j)
            iters += 1
        jobs = _waitJobs_(jobs)    
        print "Iteration "+str(iters)+" from "+str(train_images)+" completed"
        f.write("\nIteration "+str(iters)+" from "+str(train_images)+" completed")
        f.flush()

    f.write("\n*****Saving Validation******\n")
    print "*****Saving Validation******"
    iters = 0
    jobs = []
    for i in range(val_images/max_memory_images):
        val_batch, paths, labels = LOADER_VAL.load_data(verbose = 0)
        if LOADER_VAL.epoch > 0: break
        for j in range(val_batch.shape[0]):
            j = threading.Thread(target=save_worker, args=(val_batch[j], labels[j], '../datasets/places/val/'+str(iters)+'.png'))
            j.start()
            jobs.append(j)
            iters += 1
        jobs = _waitJobs_(jobs)
        print "Iteration "+str(iters)+" from "+str(val_images)+" completed"
        f.write("\nIteration "+str(iters)+" from "+str(val_images)+" completed")
        f.flush()
    print "Finished"
    f.close()

if __name__ == "__main__":
    main()

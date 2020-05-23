import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans as KMeans
from time import time
from sklearn.preprocessing import Normalizer
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
warnings.filterwarnings("ignore")

class SiftExtraction:
    
    @staticmethod
    # takes all images and convert them to grayscale. 
    # return a dictionary that holds all images category by category. 
    def load_images_from_folder(folder,class_pred):
        paths=list() #inizializziamo la lista dei path
        y=list() #inizializziamo la lista delle etichette
        classes= os.listdir(folder)
        classes_idx=range(len(classes))
        images = {}
        for filename in os.listdir(folder):
            category = []
            path = folder + "\\" + filename
            if (filename == class_pred):
                for cat in os.listdir(path):
                    img = cv2.imread(path + "\\" + cat,0)
                    #img = cv2.resize(img, (80,60))
                    if img is not None:
                        category.append(img)
                        y.append(classes.index(filename))
                        paths.append(path + "\\" + cat)
                images[filename] = category
        return images, paths, y
    
    @staticmethod
    # Creates descriptors using sift 
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    def sift_features(images):
        sift_vectors = {}
        descriptor_list = []
        descriptor_list_append = []
        sift = cv2.xfeatures2d.SIFT_create()
        for key,value in images.items():
            features = []
            for img in value:
               # print(key)
                kp, des = sift.detectAndCompute(img,None)
                if len(kp) < 1:
                   # print("nessun descrittore sift")
                    no_kp = np.zeros((1, sift.descriptorSize()), np.float32)
                    descriptor_list.extend(no_kp)
                    descriptor_list_append.append(no_kp)
                    features.append(no_kp)
                else:
                    #print(len(des))
                    descriptor_list.extend(des)
                    descriptor_list_append.append(des)
                    features.append(des)
            sift_vectors[key] = features
        return [descriptor_list, descriptor_list_append, sift_vectors]
    
    @staticmethod
    def create_cluster(num_centroidi, sift_descrptors):
        #inizializziamo l'oggetto "KMeans" impostando il numero di centroidi
        kmeans = KMeans(num_centroidi)
        #avviamo il kmeans sulle feature estratte
        start_time=time()
        kmeans.fit(sift_descrptors)
        end_time=time()
        elapsed_time=end_time-start_time
        print ("Total time: {0:0.2f} sec.".format(elapsed_time))
        print(kmeans.cluster_centers_.shape)
        return kmeans
    
    @staticmethod
    def describe_dataset(descriptor_list,kmeans):
        X=list() #inizializziamo la lista delle osservazioni
        
        for descriptor in descriptor_list:
           # print(len(descriptor))
            assignments= kmeans.predict(descriptor)
            bovw_representation, _ = np.histogram(assignments, bins=1250, range=(0,1249))
           # print(bovw_representation)
            X.append(bovw_representation)
        return X
       
    
    @staticmethod 
    def query_image(path,tree,idf,path_training,kmeans):
        X=list()
        img = cv2.imread(path,0)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        if len(kp) < 1:
            # print("nessun descrittore sift")
            des = np.zeros((1, sift.descriptorSize()), np.float32)
            #descriptor_list_append.append(no_kp)
        assignments= kmeans.predict(des)
        bovw_representation, _ = np.histogram(assignments, bins=1250, range=(0,1249))
        X.append(bovw_representation)
        X_test_tfidf=X*idf
        norm = Normalizer(norm='l2')
        X_test_tfidf_l2 = norm.transform(X_test_tfidf)
        distance, closest_idx = tree.query(X_test_tfidf_l2,k=10)
        closest_im = []
        
         
        for dist in distance[0]:
            print(dist)
        
        for indice in closest_idx[0]:
            #print(indice)
            closest_im.append(path_training[indice])
        return closest_im

def sift_extraction_bow (classe_predetta,img_query,dir_dataset,dirImgOut):
    images, paths, y_training= SiftExtraction.load_images_from_folder(dir_dataset,classe_predetta)
    
    sifts = SiftExtraction.sift_features(images)

    # Takes the descriptor list which is unordered one
    descriptor_list = sifts[0] 

    descriptor_list_append = sifts[1] 

    #concateno verticalmente tutti i descrittori 
    concatenated_features=np.vstack(descriptor_list)

    # Takes the sift features that is seperated class by class for train data
    all_bovw_feature = sifts[2] 

    centroidi = SiftExtraction.create_cluster(1250, concatenated_features)

    X = SiftExtraction.describe_dataset(descriptor_list_append,centroidi)

    #binarizziamo il vettore di rappresentazioni X_training
    #otterremo una matrice n x 500 in cui l'elemento x_ij
    #indica se la parola j-esima era presente nell'immagine i-esima
    X=np.vstack(X)
    presence = (X>0).astype(int)
    #sommiamo tutte le righe (asse "0" della matrice)
    #ogni elemento del vettore risultate indicherà il numero di
    #righe (=immagini) in cui la parola visuale era presente
    df = presence.sum(axis=0)

    #otteniamo prima il numero di immagini
    n=len(X)
    #calcoliamo il termine secondo la formula riportata prima
    idf = np.log(float(n)/(1+df))

    X_training_tfidf=X*idf

    norm = Normalizer(norm='l2')

    X_training_tfidf_l2 = norm.transform(X_training_tfidf)

    tree = KDTree(X_training_tfidf_l2)

    closest_im = SiftExtraction.query_image(img_query,tree,idf,paths,centroidi)

    print(closest_im)

    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 5
    rows = 2
    plt.title('SIFT Results')
    for i in range(1, columns*rows +1):
        img = cv2.imread(closest_im[i-1])
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    
    plt.savefig(dirImgOut+'\\SIFT_result.jpg')
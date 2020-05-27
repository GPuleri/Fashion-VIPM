# import the necessary packages
from features_extr.color.ColorDescriptor import ColorDescriptor
from features_extr.color.Searcher import Searcher
import cv2, glob, warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
    
def color (dir_class, img_query, dir_dataset, dirImgOut):
    # initialize the color descriptor
    cd = ColorDescriptor((16, 24, 3))
    """
    # open the output index file for writing
    output = open('features_extr/color/descriptor/index_' + dir_class + '_dataset.csv', "w")

    # use glob to grab the image paths and loop over them
    for imagePath in glob.glob('C:\\Users\\fabio\\Downloads\\dataset_category\\dataset_category\\' + dir_class + "/*.jpg"):
        # extract the image ID (i.e. the unique filename) from the image
        # path and load the image itself
        imageID = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath)

        # describe the image
        features = cd.describe(image)

        # write the features to file
        features = [str(f) for f in features]
        output.write("%s,%s\n" % (imageID, ",".join(features)))

    # close the index file
    output.close()
    """

    # load the query image and describe it
    query = cv2.imread(img_query)
    features = cd.describe(query)

    # perform the search
    searcher = Searcher('features_extr/color/descriptor/index_' + dir_class + '_dataset.csv')
    results = searcher.search(features)

    fig = plt.figure(figsize = (8, 8))
    columns = 5
    rows = 2
    i = 1
    plt.axis('off')
    f, axarr = plt.subplots(2, 5)
    idx=0
    f.suptitle('Color Results')
    for i in range(rows):
        for j in range (columns):
            score, resultID = results[idx]
            img = cv2.imread(resultID)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axarr[i,j].imshow(img)
            axarr[i,j].axis('off')
            axarr[i,j].set_title(idx+1)
            idx+=1
    plt.savefig(dirImgOut + '/color_result.jpg')
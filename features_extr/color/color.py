# import the necessary packages
from features_extr.color.ColorDescriptor import ColorDescriptor
from features_extr.color.Searcher import Searcher
import cv2, glob, warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
    
def color (dir_class, img_query, dir_dataset, dirImgOut):
    # initialize the color descriptor
    cd = ColorDescriptor((16, 24, 3))
    
    # open the output index file for writing
    output = open('features_extr/color/descriptor/index_' + dir_class + '_dataset.csv', "w")

    # use glob to grab the image paths and loop over them
    for imagePath in glob.glob('C:\\Users\\pule\\Documents\\dataset_category\\' + dir_class + "/*.jpg"):
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
    plt.title('Color Results')
    plt.axis('off')
    # loop over the results
    for (score, resultID) in results:
        # load the result image and display it
        img = cv2.imread(resultID)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, i, title = "Score=%s" % round(score,2))
        plt.imshow(img)
        i += 1

    plt.savefig(dirImgOut + '/color_result.jpg')
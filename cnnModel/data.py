from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATAPATH = "/Users/vincirist/Documents/HackerMaterial/pythonWS/hrobml/SimonPySlam/cnnModel/minibotSampleImgs/"
wallIm = Image.open(DATAPATH + "wall.png")
symbolNames = [ "wall", # nothing
                "arrow_left", "arrow_down", "arrow_right", "arrow_up", # arrows
                "bigS", "bigT", "bigX", # letters
                "circle", "square", # shape
                "skull", "fire", "henrik"] # funny things
symList = [ Image.open(DATAPATH + filename + ".png") for filename in symbolNames]

def addSymbol(bgImg, symImg, newSize=(80,60)):
    im = np.array(bgImg.copy().resize(newSize)).astype(np.uint8)[:,:,:3]
    imHeigth, imWidth, _ = im.shape     # WARNING: in array height and width are switched
    left   = abs( int(imWidth  * np.random.normal(.2, 0.05) ))
    top    = abs( int(imHeigth * np.random.normal(.2, 0.05) ))
    right  = min( int(imWidth  * np.random.normal(.8, 0.05) ), imWidth)
    bottom = min( int(imHeigth * np.random.normal(.8, 0.05) ), imHeigth)
    w = abs(right - left)
    h = abs(bottom - top)
    sym = np.array(symImg.copy().resize((w,h))).astype(np.uint8)

    def isBlack(rgba):
        if rgba.shape != (4,):  # catch empty values
            return False
        r, g, b, a = rgba
        tresh = 215
        return (r<tresh and g<tresh and b<tresh and a>200)

    for x in range(w):
        for y in range(h):
            color = sym[y, x]    # WARNING: x, y are also switched
            if isBlack(color):
                im[top+y, left+x] = color[:3] # only RGB (not alpha)
    return im

def createDataSet(bgImg, symImgList, amount=20, size=(80,60)):
    numOfClasses = len(symImgList)
    ims   = np.zeros((amount, size[1], size[0], 3)).astype(np.float)
    y     = np.zeros((amount, )).astype(np.int)
    # numOfTrainSamples = int( testToTrainRatio*amount +0.5)
    for i in range(amount):
        pick = np.random.randint(numOfClasses)
        symImage = symImgList[pick]
        ims[i,:,:,:] = addSymbol(bgImg, symImage, newSize=size)/255
        y[i] = pick
    return ims, y
    
def createTrainingSet(amount, trainingRatio=0.8):
    images, targets = createDataSet(wallIm, symList, amount=amount)
    # split set into trainign and testing data
    numOfTrain = int(amount * trainingRatio + .5)
    return images[:numOfTrain], targets[:numOfTrain], images[numOfTrain:], targets[numOfTrain:]

def nameOf(target):
    return symbolNames[target]


if __name__ == "__main__":
    ims, cla = createDataSet(wallIm, symList)
    for i, c in zip(ims, cla):
        plt.imshow(i)
        plt.title(nameOf(c))
        plt.show()
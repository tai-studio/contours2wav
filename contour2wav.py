import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

def img2contours(img, thresh = 127, blur = 3):
    """ convert image to contours
    """

    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray_img, (blur, blur), 0)
    ret, threshed = cv.threshold(blurred, thresh, 255, 0)
    # find contour
    contours, hierarchy = cv.findContours(threshed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours

def contours2centers(contours):
    """ calculate centers of contours"""
    def centerOf(contour):
        """ calculate center of contour
        """
        x, y = contour.reshape((-1, 2)).transpose()
        return np.array([np.mean(x), np.mean(y)])
    
    centers = list(map(centerOf, contours))
    return centers

def contours2polarDistances(contours, centers):
    """ convert contours to polar coordinates relative to center
    """
    def contour2polarDistances(contour, center):
        """ convert contours to polar coordinates relative to center
        returns: (phis, distances)
        """
        x, y = (contour.reshape((-1, 2)) - center).transpose()
        phi = np.arctan2(y,x) # angle in radians
        rho = np.sqrt(x**2+y**2) # distance from center
        return np.array([phi, rho]).transpose()

    polarContours = list(map(contour2polarDistances, contours, centers))
    return polarContours

def contours2centeredContours(contours, centers):
    """ convert contours to contours centered around their centers
    """
    def contour2centeredContour(contour, center):
        """ convert contours to contours centered around their centers
        """
        return contour - center

    centeredContours = list(map(contour2centeredContour, contours, centers))
    return centeredContours

def resampleSignal(signal, n):
    """ resample signal to n samples
    signal may be 1D or 2D
    """
    if len(signal.shape) == 1:
        return np.interp(np.linspace(0, len(signal), n),
                         np.arange(len(signal)), signal)
    else:
        return np.array([resampleSignal(s, n) for s in signal.transpose()]).transpose()

def savecontours2csvs(contours, dirname, numSamples=100):
    """ save contours to csv files"""
    
    def savecontour2csv(contour, idx):
        signal = contour.reshape((-1, 2))
        # resample signal to numSamples
        signal = resampleSignal(signal, numSamples)
        
        np.savetxt(f"{dirname}/c_{idx}.csv",
                   signal, delimiter=",")

    list(map(savecontour2csv, contours, range(len(contours))))

def savecontours2wav(contours, dirname, numSamples=100, sampleRate=48000, center=True, startIdx=0):
    """ save contours to wav files"""
    
    def savecontour2wav(contour, idx):
        signal = contour.reshape((-1, 2))
        # resample signal to numSamples
        signal = resampleSignal(signal, numSamples)
        # normalize signal
        if center:
            signal = signal - np.mean(signal)

        signal = signal / np.max(np.abs(signal))

        wavfile.write(f"{dirname}/cx_{idx}_{numSamples}at{sampleRate}.wav", sampleRate, (signal.transpose()[0]).astype(np.float32))
        wavfile.write(f"{dirname}/cy_{idx}_{numSamples}at{sampleRate}.wav", sampleRate, (signal.transpose()[1]).astype(np.float32))

    list(map(savecontour2wav, contours, range(startIdx, len(contours) + startIdx)))



def savepolars2csvs(polarContours, dirname, numSamples=100):
    """ save polar contours to csv files"""

    def savepolar2csv(polarContour, idx):
        signal = polarContour.transpose()[1]
        # resample signal to numSamples
        signal = resampleSignal(signal, numSamples)
        np.savetxt(f"{dirname}/p_{idx}.csv",
                   signal, delimiter=",")

    list(map(savepolar2csv, polarContours, range(len(polarContours))))

def savepolars2wav(polarContours, dirname, numSamples=100, sampleRate=48000, center=True, startIdx=0):
    """ save polar contours to wav files"""

    def savepolar2wav(polarContour, idx):
        signal = polarContour.transpose()[1]
        # resample signal to numSamples
        signal = resampleSignal(signal, numSamples)

        # normalize signal
        if center:
            signal = signal - np.mean(signal)
        signal = signal / np.max(np.abs(signal))

        wavfile.write(f"{dirname}/px_{idx}_{numSamples}at{sampleRate}.wav", sampleRate, signal.astype(np.float32))

    list(map(savepolar2wav, contours, range(startIdx, len(contours) + startIdx)))

if __name__ == '__main__':
    """
    extract contours from image and save to wav files
    """
    import argparse
    import os

    # parse command line arguments for input file name
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--minPoints', nargs=1, type=int, default=[100], help='minimum number of points in contour (default: 100)')
    parser.add_argument('-n', '--maxPoints', nargs=1, type=int, default=[1000], help='maximum number of points in contour (default: 1000)')
    parser.add_argument('-t', '--threshold', nargs=1, type=int, default=[127], help='threshold for binarization (default: 127)')
    parser.add_argument('-s', '--numSamples', nargs=1, type=int, default=[8192], help='number of samples per contour (default: 8192)')
    parser.add_argument('-r', '--sampleRate', nargs=1, type=int, default=[48000], help='sample rate (default: 48000)')
    parser.add_argument('-i', '--startIdx', nargs=1, type=int, default=[0], help='start index for contour numbering (default: 0)')
    parser.add_argument('-p', '--polar', action='store_true', help='save polar coordinates')
    parser.add_argument('infile', nargs=1, help='input image file [.jpg, .png, ...]')
    parser.add_argument('outdir', nargs=1, help='results will be saved to <outdir>/<infile-basename>/<n>.wav')
    args = parser.parse_args()

    infile = args.infile[0]
    outdir = args.outdir[0]
    minPoints = args.minPoints[0]
    maxPoints = args.maxPoints[0]
    threshold = args.threshold[0]
    numSamples = args.numSamples[0]
    sampleRate = args.sampleRate[0]
    startIdx = args.startIdx[0]
    writePolar = args.polar
    # read image
    img = cv.imread(infile)

    contours = img2contours(img, thresh=threshold, blur=3)
    
    # filter out contours with less than minPoints points
    contours = list(filter(lambda c: len(c) > minPoints, contours))
    # filter out contours with more than maxPoints points
    contours = list(filter(lambda c: len(c) < maxPoints, contours))

    # resample contours to be closed
    def resampleContour(contour):
        # epsilon = 0.001*cv.arcLength(contour, closed=True)
        epsilon = 0
        return cv.approxPolyDP(contour, epsilon=epsilon, closed=True)

    # compute features
    contours = list(map(lambda c: resampleContour(c), contours))
    centers = contours2centers(contours)

    print(f"{infile}:\tcontours found: {len(contours)}")

    cartesianContours = contours2centeredContours(contours, centers)



    # create output directory if it doesn't exist
    if not os.path.exists(outdir[0]):
        os.makedirs(outdir[0])

    infileprefix = infile.rpartition("/")[-1].rpartition(".")[0]
    dirname = f"{outdir}/{infileprefix}"

    # create file directory if it doesn't exist
    if not os.path.exists(dirname):
        os.makedirs(dirname)


    # save to csv files
    # savepolars2csvs(polarContours, dirname)
    # savecontours2csvs(contours, dirname)

    # save to wav files
    savecontours2wav(cartesianContours, dirname, numSamples, sampleRate, center=False, startIdx=startIdx)

    if writePolar:
        polarContours = contours2polarDistances(contours, centers)
        savepolars2wav(polarContours, dirname, numSamples, sampleRate, center=False, startIdx=startIdx)
    

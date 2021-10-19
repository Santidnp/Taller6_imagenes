#Elaborado por Laura Juliana Ramos , Santiago Garcia

#Importar librerias
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

# seleccione la identificación de la imagen entre  'sift', 'surf', 'brisk', 'orb'
cv2.ocl.setUseOpenCL(False)
feature_extractor = 'orb' #  'sift', 'surf', 'brisk', 'orb'
feature_matching = 'bf'

if __name__ == '__main__':
    #Cargar las imagenes de referencia y otra de ellas (la de referencia es la imagen2 )
    image_name_1 =  r'C:\Users\Laura\Desktop\Procesamiento imagenes\stitching\Image_1.jpeg'
    image_name_2 = r'C:\Users\Laura\Desktop\Procesamiento imagenes\stitching\Image_2.jpeg'


    #La imagen de train es la imagen a transformar
    trainImg = cv2.imread(image_name_2)
    trainImg = cv2.resize(trainImg, (600, 600), interpolation=cv2.INTER_CUBIC)
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)

    queryImg = cv2.imread(image_name_1)
    queryImg = cv2.resize(queryImg, (600, 600), interpolation=cv2.INTER_CUBIC)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)

    #MOSTRAR LAS IMAGENES 1 Y 2
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(queryImg, cmap="gray")
    ax1.set_xlabel("Imagen referencia", fontsize=14)

    ax2.imshow(trainImg, cmap="gray")
    ax2.set_xlabel("Imagen a transformar ", fontsize=14)

    plt.show()

    def detectAndDescribe(image, method=None):

        assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
        # Calcule puntos clave y descriptores de características utilizando un método específico
        if method == 'sift':
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif method == 'surf':
            descriptor = cv2.xfeatures2d.SURF_create()
        elif method == 'brisk':
            descriptor = cv2.BRISK_create()
        elif method == 'orb':
            descriptor = cv2.ORB_create()
        # get keypoints and descriptors
        (kps, features) = descriptor.detectAndCompute(image, None)
        return (kps, features)

    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)

    # mostrar los puntos clave y las características detectadas en ambas imágenes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(trainImg_gray, kpsA, None, color=(0, 255, 0)))
    ax1.set_xlabel("(a)", fontsize=14)
    ax2.imshow(cv2.drawKeypoints(queryImg_gray, kpsB, None, color=(0, 255, 0)))
    ax2.set_xlabel("(b)", fontsize=14)

    plt.show()

    def createMatcher(method, crossCheck):
        if method == 'sift' or method == 'surf':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        elif method == 'orb' or method == 'brisk':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        return bf

    def matchKeyPointsBF(featuresA, featuresB, method):
        bf = createMatcher(method, crossCheck=True)
        # Descriptores de coincidencias.
        best_matches = bf.match(featuresA, featuresB)
        # Ordene las características en orden de distancia.
        # Los puntos con poca distancia (más similitud) se ordenan primero en el vector
        rawMatches = sorted(best_matches, key=lambda x: x.distance)
        print("Raw matches (Brute force):", len(rawMatches))
        return rawMatches


    def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
        bf = createMatcher(method, crossCheck=False)
        # calcular las coincidencias sin procesar e inicializar la lista de coincidencias reales
        rawMatches = bf.knnMatch(featuresA, featuresB, 2)
        print("Raw matches (knn):", len(rawMatches))
        matches = []

        for m, n in rawMatches:

            if m.distance < n.distance * ratio:
                matches.append(m)
        return matches


    fig = plt.figure(figsize=(20, 8))

    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches[:100],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, np.random.choice(matches, 100),
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3)
    plt.show()

    def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
        # convertir los puntos clave en matrices numpy
        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])

        if len(matches) > 4:

            # construye los dos conjuntos de puntos
            ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
            ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

            # estimar la homografía entre los conjuntos de puntos
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            return (matches, H, status)
        else:
            return None


    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    if M is None:
        print("Error!")
    (matches, H, status) = M
    print(H)
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

    plt.imshow(result)
    plt.show()

    # transformar la imagen panorámica a escala de grises y establecer un umbral
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Encontrar contornos de la imagen binaria
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Obtener el área de contorno máxima
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    # recorta la imagen a las coordenadas
    result = result[y:y + h, x:x + w]

    # mostrar la imagen recortada
    plt.imshow(result)
    plt.show()

#########################UNION 3 imagenes##############################

    image_name_1= r'C:\Users\Laura\Desktop\Procesamiento imagenes\stitching\Image_3.jpeg'
    trainImg =cv2.imread(image_name_1)
    trainImg = cv2.resize(trainImg, (600, 600), interpolation=cv2.INTER_CUBIC)
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)

    queryImg = result
    queryImg = cv2.resize(queryImg, (600, 600), interpolation=cv2.INTER_CUBIC)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16, 9))
    ax1.imshow(queryImg, cmap="gray")
    ax1.set_xlabel("Query image", fontsize=14)

    ax2.imshow(trainImg, cmap="gray")
    ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)

    plt.show()

    def detectAndDescribe(image, method=None):
        assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
        # detect and extract features from the image
        if method == 'sift':
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif method == 'surf':
            descriptor = cv2.xfeatures2d.SURF_create()
        elif method == 'brisk':
            descriptor = cv2.BRISK_create()
        elif method == 'orb':
            descriptor = cv2.ORB_create()
        # get keypoints and descriptors
        (kps, features) = descriptor.detectAndCompute(image, None)
        return (kps, features)

    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)

    # display the keypoints and features detected on both images
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(trainImg_gray, kpsA, None, color=(0, 255, 0)))
    ax1.set_xlabel("(a)", fontsize=14)
    ax2.imshow(cv2.drawKeypoints(queryImg_gray, kpsB, None, color=(0, 255, 0)))
    ax2.set_xlabel("(b)", fontsize=14)

    plt.show()

    def createMatcher(method, crossCheck):
        if method == 'sift' or method == 'surf':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
        elif method == 'orb' or method == 'brisk':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
        return bf

    def matchKeyPointsBF(featuresA, featuresB, method):
        bf = createMatcher(method, crossCheck=True)
        # Match descriptors.
        best_matches = bf.match(featuresA, featuresB)
        # Sort the features in order of distance.
        # The points with small distance (more similarity) are ordered first in the vector
        rawMatches = sorted(best_matches, key=lambda x: x.distance)
        print("Raw matches (Brute force):", len(rawMatches))
        return rawMatches


    def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
        bf = createMatcher(method, crossCheck=False)
        # compute the raw matches and initialize the list of actual matches
        rawMatches = bf.knnMatch(featuresA, featuresB, 2)
        print("Raw matches (knn):", len(rawMatches))
        matches = []

        # loop over the raw matches
        for m, n in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * ratio:
                matches.append(m)
        return matches


    fig = plt.figure(figsize=(20, 8))

    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches[:100],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, np.random.choice(matches, 100),
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3)
    plt.show()

    def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
        # convert the keypoints to numpy arrays
        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])

        if len(matches) > 4:

            # construct the two sets of points
            ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
            ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

            # estimate the homography between the sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            return (matches, H, status)
        else:
            return None


    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    if M is None:
        print("Error!")
    (matches, H, status) = M
    print(H)
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

    plt.imshow(result)
    plt.show()

    # transform the panorama image to grayscale and threshold it
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

        # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
    result = result[y:y + h, x:x + w]

        # show the cropped image

    plt.imshow(result)
    plt.show()

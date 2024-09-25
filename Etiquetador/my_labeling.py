__authors__ = ['1673129', '1671727', '1604517']
__group__ = '8'

import random
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter

from KNN import *
from Kmeans import *

import utils
from utils_data import *


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/',
                                         gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    """
    --------------------------------------------------------------------FUNCTIONS--------------------------------------------------------------------
    """

    def unique_colors_percentages(colors, percentages):
        """
        Args:
            colors: list of color names obtained from the kmeans object
            percentages: list of percentages

        Return:
            Return the filtered color_list (removed duplicates,
                                          removed irrelevant ...)
                 and the percentages of the colors
        """

        unique_colors = np.unique(colors)

        unique_percentages = []
        for color in utils.colors:
            actual_color_percentages = []
            for c, p in zip(colors, percentages):
                if c == color:
                    actual_color_percentages.append(p)
            if actual_color_percentages:
                actual_color_percentages = np.array(
                    [actual_color_percentages]).mean()
            unique_percentages.append(actual_color_percentages)

        return unique_colors, unique_percentages


    def Retrieval_by_color(list_images, predicted_colors, predicted_scores, search, n):
        """
        Args:
            list_images: dataset of the images
            predicted_colors: list of the colors we have obtained after applying
                                the K-means
            predicted_scores: list of the percentages obtained
            search: colors we want to search in the images
            n: the number of images we want to print

        Return:
            Return a list of the indexes that have these colors
        """

        if type(search) != list: search = [search]

        # agafem els indexs de les imatges que tenen TOTS els colors de search
        indicesToPrint = [i for i, colors in enumerate(predicted_colors) if
                          all(color in colors for color in search)]

        # agafem els scores de nomes les imatges que ens interessen
        scores = predicted_scores[indicesToPrint]

        # llista amb les etiquetes de colors de les imatges que ens interessen
        indicesColors = predicted_colors[indicesToPrint]

        # llista amb els indexs dels colors que li hem passat a search
        indicesSearch = [i for i, color in enumerate(utils.colors) if
                         color in search]

        # per a cada imatge, fem MEAN dels percentatges dels colors que ens demanen a search
        scores_mean = np.mean(scores[:, indicesSearch], axis=1)

        # ordenem indicesToPrint i indicesColors en funció de scores_mean
        ordenat = sorted(zip(indicesToPrint, indicesColors, scores_mean),
                         key=lambda x: -x[2])

        # guardem indicesToPrint i indicesColors
        indicesToPrint = [x[0] for x in ordenat]

        # agafem les imatges ordenades
        imagesGiven = list_images[indicesToPrint]

        # fem el visualize_retrieval
        visualize_retrieval(imagesGiven, n)

        return indicesToPrint


    def Retrieval_by_shape(list_images, predicted_shapes, predicted_scores, search, n):
        """
        Args:
            list_images: dataset of the images
            predicted_shapes: list of the shapes we have obtained after applying
                                the KNN
            predicted_scores: list of the percentages obtained
            search: shapes we want to search in the images
            n: the number of images we want to print

        Return:
            Return a list of the indexes that have these shapes
        """

        indicesToPrint = []
        indicesShapes = []

        for index, element in enumerate(predicted_shapes):
            if element in search:
                indicesToPrint.append(index)
                indicesShapes.append(element)

        ordenat = sorted(zip(indicesToPrint, indicesShapes,
                             predicted_scores[indicesToPrint]), key=lambda x: -x[2])

        indicesToPrint = [x[0] for x in ordenat]

        imagesGiven = list_images[indicesToPrint]

        visualize_retrieval(imagesGiven, n)

        return indicesToPrint


    def Retrieval_combined(list_images, predicted_shapes,
                           predicted_shape_scores, search_shape,
                           predicted_colors, predicted_color_scores,
                           search_color, n):
        """
        Args:
            list_images: dataset of the images
            predicted_shapes: list of the shapes we have obtained after applying
                                the KNN
            predicted_shape_scores: list of the percentages obtained (shapes)
            search_shape: shapes we want to search in the images
            predicted_colors: list of the colors we have obtained after applying
                                the K-means
            predicted_color_scores: list of the percentages obtained (colors)
            search_color: colors we want to search in the images
            n: the number of images we want to print

        Return:
            Return a list of the indexes that have these colors and shapes
        """

        indexes_color = Retrieval_by_color(list_images, predicted_colors,
                                          predicted_color_scores, search_color,
                                          n)
        indexes_shape = Retrieval_by_shape(list_images, predicted_shapes,
                                          predicted_shape_scores, search_shape,
                                          n)

        intersection = list(set(indexes_color) & set(indexes_shape))
        images_to_print = list_images[intersection]

        visualize_retrieval(images_to_print, n)

        return intersection


    def Kmean_statistics(Kmeans, Kmax):
        """
        Args:
            Kmeans: algorithm Kmeans after applying it
            Kmax: maximum number of k possible

        Return:
            Return a list of all the values calculated
        """

        wcd = []
        iterations = []
        times = []
        icd = []
        values = []

        while Kmeans.K < Kmax:
            start_time = time.time()
            Kmeans.fit()

            # calcul del temps
            actual_time = time.time() - start_time
            times.append(actual_time)

            # calcul de WCD
            Kmeans.withinClassDistance()
            wcd.append(Kmeans.WCD)

            # calcul de les iterations
            iterations.append(Kmeans.num_iter)
            Kmeans.interClassDistance()

            # calcul de ICD
            icd.append(Kmeans.ICD)
            values.append(Kmeans.K)

            Kmeans.K += 1

        # Plot dels resultats obtinguts

        # WCD vs. K
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.plot(values, wcd, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Within-Class Distance (WCD)')
        plt.title('WCD vs. K')

        # iterations vs. K
        plt.subplot(2, 2, 2)
        plt.plot(values, iterations, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Number of Iterations')
        plt.title('Iterations vs. K')

        # temps vs. K
        plt.subplot(2, 2, 3)
        plt.plot(values, times, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Time')
        plt.title('Time vs. K')

        # ICD vs. K
        plt.subplot(2, 2, 4)
        plt.plot(values, icd, marker='o')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Intra-Class Distance (ICD)')
        plt.title('ICD vs. K')

        plt.tight_layout()
        plt.show()

        return wcd, iterations, times, icd, values


    def Get_color_accuracy(kmeans_results, ground_truth):
        """
        Args:
            kmeans_results: results of the algorithm Kmeans after applying it
            ground_truth: expected colors

        Return:
            Return the accuracy calculated
        """

        total = 0
        for a, b in zip(kmeans_results, list(ground_truth)):
            rao = 1 / len(b)
            if sorted(a) == sorted(b):
                total += 1
            else:
                for x in b:
                    if x in a:
                        total += rao
        return 100 * (total / len(kmeans_results))


    def Get_shape_accuracy(labels, ground_truth):
        """
        Args:
            labels: results of predicted values after applying the algorithm KNN
            ground_truth: expected shapes

        Return:
            Return the accuracy calculated
        """

        return np.mean(labels == ground_truth) * 100


        """
        creem una nova llista i per cada imatge apliquem el metode rgb2gray i l'afegim a la llista nova
        
        """
    def image_to_grey(images):
        """
        Args:
            images: the list of images we want to change to color grey

        Return:
            Return the list of images transformed to grey

        """

        GreyImages = []
        for image in images:
            GreyImages.append(utils.rgb2gray(image))

        return np.array(GreyImages)

    """
    TESTS
    """

    """
    TEST: Retrieval_by_color, K-means
    """

    retrieval_by_color = False

    if retrieval_by_color:
        start_time = time.time()

        best_color_labels = []
        best_color_percentages = []
        for image in imgs:
            km = KMeans(image, K=4, options={'km_init': 'first', 'fitting': 'WCD'})
            km.find_bestK(11, 0.2)

            # Fem la predicció de colors i percentages, utilitzant el K-means
            color_labels = get_colors(km.centroids)
            colorProb = utils.get_color_prob(km.centroids)
            color_percentages = np.max(colorProb, axis=1)

            # color_labels sense repeticions (array)
            unique_color_labels, unique_color_percentages = \
                unique_colors_percentages(color_labels, color_percentages)

            best_color_labels.append(unique_color_labels)
            best_color_percentages.append(unique_color_percentages)

        best_color_labels = np.array(best_color_labels, dtype=object)
        best_color_percentages = np.array(best_color_percentages, dtype=object)

        colors = ["Black"]
        Retrieval_by_color(imgs, best_color_labels, best_color_percentages, colors, 8)

        print("Retrieval_by_color time:", time.time() - start_time)

    """
    TEST: Retrieval_by_shape, KNN
    """

    retrieval_by_shape = False

    if retrieval_by_shape:
        start_time = time.time()

        knn = KNN(np.array(train_imgs), train_class_labels)
        predicted_class_labels = knn.predict(imgs, 5)

        predicted_class_percentages = np.array([])
        for fila in knn.neighbors:
            unique, counts = np.unique(fila, return_counts=True)
            unique, counts = unique[0], counts[0]
            max_counts = np.max(counts)
            predicted_class_percentages = np.append(predicted_class_percentages,
                                                    max_counts / fila.size)

        shapes = ["Flip Flops"]
        Retrieval_by_shape(imgs, predicted_class_labels, predicted_class_percentages, shapes, 8)

        print("Retrieval_by_shape time:", time.time() - start_time)

    """
    TEST: Retrieval_combined, K-means and KNN
    """

    retrieval_combined = False

    if retrieval_combined:
        start_time = time.time()

        # Inicialitzem knn
        knn = KNN(np.array(train_imgs), train_class_labels)
        predicted_class_labels = knn.predict(imgs, 5)

        predicted_class_percentages = np.array([])
        for fila in knn.neighbors:
            unique, counts = np.unique(fila, return_counts=True)
            unique, counts = unique[0], counts[0]
            max_counts = np.max(counts)
            predicted_class_percentages = np.append(predicted_class_percentages,
                                                    max_counts / fila.size)

        # Inicialitzem Kmeans
        best_color = []
        best_color_percentages = []
        for image in imgs:
            km = KMeans(image, K=4, options={'km_init': 'first', 'fitting': 'WCD'})
            km.find_bestK(11, 0.2)

            color_labels = get_colors(km.centroids)
            colorProb = utils.get_color_prob(km.centroids)
            color_percentages = np.max(colorProb, axis=1)

            unique_color_labels, unique_color_percentages = \
                unique_colors_percentages(color_labels, color_percentages)

            best_color.append(unique_color_labels)
            best_color_percentages.append(unique_color_percentages)

        best_color = np.array(best_color, dtype=object)
        best_color_percentages = np.array(best_color_percentages, dtype=object)

        colors = ["Blue"]
        shapes = ["Jeans"]
        Retrieval_combined(imgs, predicted_class_labels,
                           predicted_class_percentages,
                           shapes, best_color, best_color_percentages, colors,
                           8)

        print("Retrieval_combined time:", time.time() - start_time)

    """
    TEST: Kmean_statistics
    """

    kmean_statistics = False

    if kmean_statistics:
        start_time = time.time()

        best_color_labels = []
        best_color_percentages = []

        for image in imgs:
            km = KMeans(image, K=4, options={'km_init': 'first', 'fitting': 'WCD'})
            km.fit()

        wcd, iterations, times, icd, values = Kmean_statistics(km, 11)

        print("Kmean_statistics time:", time.time() - start_time)

    """
    TEST: Color Accuracy
    """

    color_accuracy = False

    if color_accuracy:
        start_time = time.time()

        best_color_labels = []
        best_color_percentages = []
        for image in imgs:
            km = KMeans(image, K=4, options={'km_init': 'first', 'fitting': 'WCD'})
            km.find_bestK(11, 0.2)

            color_labels = get_colors(km.centroids)
            colorProb = utils.get_color_prob(km.centroids)
            color_percentages = np.max(colorProb, axis=1)

            unique_color_labels, unique_color_percentages = \
                unique_colors_percentages(color_labels, color_percentages)

            best_color_labels.append(unique_color_labels)
            best_color_percentages.append(unique_color_percentages)

        best_color_labels = np.array(best_color_labels, dtype=object)
        best_color_percentages = np.array(best_color_percentages, dtype=object)

        accuracy = Get_color_accuracy(best_color_labels, test_color_labels)

        print(f'Accuracy on color prediction: {accuracy} %')

        print("Color_accuracy time:", time.time() - start_time)

    """
    TEST: Shape Accuracy
    """

    shape_accuracy = False

    if shape_accuracy:
        start_time = time.time()

        knn = KNN(np.array(test_imgs), test_class_labels)
        predicted_class_labels = knn.predict(np.array(test_imgs), 5)

        percentages = np.array([])
        for row in knn.neighbors:
            unique, counts = np.unique(row, return_counts=True)
            unique, counts = unique[0], counts[0]
            max_counts = np.max(counts)
            percentages = np.append(percentages, max_counts / row.size)
        predicted_class_percentages = percentages

        accuracy = Get_shape_accuracy(predicted_class_labels, test_class_labels)

        print(f'Accuracy on shape prediction: {accuracy} %')

        print("Shape_accuracy time:", time.time() - start_time)

    """
    TEST: init_centroids, using 4 different options['km_init']: 
        (first, random, last, extreme)
        Implemented in function _init_centroids()
        Which of these is the best? We can do the test of color accuracy to determine it
    """
    
    #CENTROIDS------------------------------------------------------------------------------------------------------------

    """
    
    Per cada centroide i per cada imatge del conjunt d'imatges crides al kmneas, al find best_k i al color accuracy
    
    
    
    
    """
    init_centroids = True

    if init_centroids:
        start_time = time.time()

        options = ['first', 'last', 'extreme']

        for option in options:

            best_color_labels = []
            best_color_percentages = []
            for image in imgs:
                km = KMeans(image, 4, options={'km_init': option, 'fitting': 'WCD'})
                km.find_bestK(11, 0.2)

                color_labels = get_colors(km.centroids)
                colorProb = utils.get_color_prob(km.centroids)
                color_percentages = np.max(colorProb, axis=1)

                unique_color_labels, unique_color_percentages = \
                    unique_colors_percentages(color_labels, color_percentages)

                best_color_labels.append(unique_color_labels)
                best_color_percentages.append(unique_color_percentages)

            best_color_labels = np.array(best_color_labels, dtype=object)
            best_color_percentages = np.array(best_color_percentages, dtype=object)

            accuracy = Get_color_accuracy(best_color_labels, test_color_labels)

            print(f'Accuracy on color prediction, using option {option}: {accuracy} %')

        print("Color_accuracy time:", time.time() - start_time)

    """
    TEST: compare_heuristics, using 3 different options['fitting']: (WCD, ICD, FC)
    """

    compare_heuristics = False

    if compare_heuristics:
        start_time = time.time()

        max_K = 11

        wcd_values = []
        icd_values = []
        fc_values = []

        for k in range(2, max_K + 1):
            kmeans = KMeans(cropped_images[0], K=k, options={'fitting': 'WCD'})
            kmeans.fit()
            kmeans.withinClassDistance()
            wcd_values.append(kmeans.WCD)

            kmeans.options['fitting'] = 'ICD'
            kmeans.fit()
            kmeans.interClassDistance()
            icd_values.append(kmeans.ICD)

            kmeans.options['fitting'] = 'FC'
            kmeans.fit()
            kmeans.withinClassDistance()
            kmeans.interClassDistance()
            kmeans.FisherCoefficient()
            fc_values.append(kmeans.FC)

        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_K + 1), wcd_values, label='WCD')
        plt.plot(range(2, max_K + 1), icd_values, label='ICD')
        plt.plot(range(2, max_K + 1), fc_values, label='FC')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Heuristic Value')
        plt.title('Comparison of Heuristics')
        plt.legend()
        plt.grid(True)
        plt.show()

        print("compare_heuristics time:", time.time() - start_time)

    """
    TEST: Find best K while changing llindar
    """

    find_bestK = False

    if find_bestK:
        start_time = time.time()

        bestKs = []

        llindars = np.arange(0.1, 1.0, 0.1)

        for llindar in llindars:
            bestK = []
            for image in imgs:
                km = KMeans(image, 4, options={'km_init': 'first', 'fitting': 'WCD'})
                k = km.find_bestK(11, llindar)
                bestK.append(k)

            counts = Counter(bestK)
            K = counts.most_common(1)[0][0]
            bestKs.append(K)

        plt.plot(llindars, bestKs, marker='o')
        plt.xlabel('Llindar')
        plt.ylabel('Best K')
        plt.title('Llindar vs. best K')
        plt.grid(True)
        plt.show()

        print("Find best K time:", time.time() - start_time)

    """
    TEST: Features KNN, using grey images
    """
    
    
        """
        cridem a la funció image to grey per canviar les imatges i despres cridem al knn i despres al retrieval by shape      
        """

    Features_KNN = False

    if Features_KNN:
        start_time = time.time()

        imgs_GreyImages = image_to_grey(imgs)
        train_imgs_GreyImages = image_to_grey(train_imgs)

        knn = KNN(np.array(train_imgs_GreyImages), train_class_labels)
        predicted_class_labels = knn.predict(imgs_GreyImages, 5)

        predicted_class_percentages = np.array([])
        for fila in knn.neighbors:
            unique, counts = np.unique(fila, return_counts=True)
            unique, counts = unique[0], counts[0]
            max_counts = np.max(counts)
            predicted_class_percentages = np.append(
                predicted_class_percentages,
                max_counts / fila.size)

        shapes = ["Flip Flops"]
        Retrieval_by_shape(imgs_GreyImages, predicted_class_labels,
                           predicted_class_percentages, shapes, 8)

        print("Features_KNN time:", time.time() - start_time)



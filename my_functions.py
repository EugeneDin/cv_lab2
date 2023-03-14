# -*- coding: utf-8 -*-
# *******************************************************************************
# 
#                               CV Lab 2
#
#                   Author: Evgeny B.A. ITMO University
# 
# _______________________________________________________________________________

import cv2
import numpy as np
import time

#Вариант без встроенной функции OpenCV (SQDIFF)
def template_matching_my(src, temp):

    pt = template_matching_my_calculate(src, temp)

    #Рисуем прямоугольник
    cv2.rectangle(src, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 2)

    #Отображаем результат
    cv2.imshow("Image",src)
    cv2.waitKey(0)
    return

def template_matching_my_calculate(src, temp):

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Получаем размеры исходного изображения
    h = gray.shape[0]
    w = gray.shape[1]

    # Получаем размеры темплата
    ht = temp.shape[0]
    wt = temp.shape[1]

    # Массив для хранения показателя метрики
    score = np.empty((h - ht, w - wt))

    # Слайдим по картинке
    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            # Алгоритм SQDIFF
            #https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dab65c042ed62c9e9e095a1e7e41fe2773
            diff = np.power((gray[dy:dy + ht, dx:dx + wt] - temp), 2)
            score[dy, dx] = diff.sum()

    pt = np.unravel_index(score.argmin(), score.shape)

    return(pt[1], pt[0])



#Вариант реализации средствами OpenCV
def template_matching_ocv(src, gray, temp):

    # Получаем параметры картинок
    h, w = temp.shape

    res = cv2.matchTemplate(gray,temp,cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(src,top_left, bottom_right, (0, 0, 200), 2)

    #Отображаем результат
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image",src)
    cv2.resizeWindow('Image', 800, 600)
    cv2.moveWindow('Image', 400, 100)
    cv2.waitKey(0)
    return

#Вариант реализации feature-matching SIFT
def feature_matching_sift(src, temp):

    # Получаем параметры картинок
    h, w = temp.shape

    # Переводим в ЧБ
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # Создаем экземпляр SIFT-детектора
    sift = cv2.SIFT_create()

    # Определяем кейпоинты и дескрипторы на оригинальном изображении
    kp1, des1 = sift.detectAndCompute(gray, None)

    # Определяем кейпоинты и дескрипторы на темплате
    kp2, des2 = sift.detectAndCompute(temp, None)

    # Создаем BFMatcher
    bf = cv2.BFMatcher()

    # Сравниваем дескрипторы
    matches = bf.match(des1, des2)

    # Сортируем по расстоянию
    matches = sorted(matches, key=lambda x: x.distance)

    # Рисуем первые 10 матчей
    #img3 = cv2.drawMatches(img, kp1, temp, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print("Matches: {0}".format(len(matches)))

    # Получаем координаты совпадений
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]

    pt = list_kp1[0]

    x = int(pt[0] - w / 2)
    y = int(pt[1] - h / 2)

    cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 200), 2)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image",src)
    cv2.resizeWindow('Image', 800, 600)
    cv2.moveWindow('Image', 400, 100)
    cv2.waitKey(0)
    return

# Функция для печати в консоль размеров изображения типа cv2
def getImageSize(resized_temp_image):
    height_image, width_image = resized_temp_image.shape
    print(height_image)
    print(width_image)
# -*- coding: utf-8 -*-
# *******************************************************************************
# 
#                               CV Lab 2
#
#                   Author: Evgeny B.A. ITMO University
# 
# _______________________________________________________________________________


import cv2
import my_functions


# --------------------------------------------------------------------------------------
#                                   
#                                           MAIN
# 
# --------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Считываем картинки
    main_image = cv2.imread('./media/set_of_objects.jpg')
    temp_image = cv2.imread('./media/temp8.jpg')

    # Конвертируем в ЧБ
    gray_image = cv2.cvtColor(main_image, cv2.COLOR_RGB2GRAY)
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)

    #Масштабируем для сохранения одного и того же размера
    scale_width = 1044
    scale_height = 834
    scale_points = (scale_width, scale_height)
    resized_temp_image = cv2.resize(temp_image, scale_points, interpolation= cv2.INTER_LINEAR)

    # Получаем параметры картинок
    # my_functions.getImageSize(resized_temp_image)

    # Вызов реализации средствами OpenCV
    my_functions.template_matching_ocv(main_image, gray_image, resized_temp_image)
    
    # Вызов реализации feature-matching SIFT
    # my_functions.feature_matching_sift(main_image, resized_temp_image)
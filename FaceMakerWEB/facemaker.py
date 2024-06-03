import os
import cv2
import numpy as np
import math

_height = 600
_width = 600


def similarity_transform(input_p, output_p):
    sin_60 = math.sin((60 * math.pi)/180)
    cos_60 = math.cos((60 * math.pi)/180)
    x_in = cos_60 * (input_p[0][0] - input_p[1][0]) - sin_60 * (input_p[0][1] - input_p[1][1]) + input_p[1][0]
    y_in = sin_60 * (input_p[0][0] - input_p[1][0]) + cos_60 * (input_p[0][1] - input_p[1][1]) + input_p[1][1]
    x_out = cos_60 * (output_p[0][0] - output_p[1][0]) - sin_60 * (output_p[0][1] - output_p[1][1]) + output_p[1][0]
    y_out = sin_60 * (output_p[0][0] - output_p[1][0]) + cos_60 * (output_p[0][1] - output_p[1][1]) + output_p[1][1]

    input_p = [[input_p[0][0], input_p[0][1]],
               [input_p[1][0], input_p[1][1]],
               [x_in, y_in]]
    output_p = [[output_p[0][0], output_p[0][1]],
                [output_p[1][0], output_p[1][1]],
                [x_out, y_out]]
    tform = cv2.estimateAffinePartial2D(np.array(input_p), np.array(output_p))
    return tform[0]


def rect_contains(rect, point):
    return not (point[0] < rect[0] or point[1] < rect[1] or point[0] > rect[2] or point[1] > rect[3])


def calculate_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))

    delaunay = []
    triangle_list = subdiv.getTriangleList()
    for triangle in triangle_list:
        point1 = (triangle[0], triangle[1])
        point2 = (triangle[2], triangle[3])
        point3 = (triangle[4], triangle[5])
        triple = [point1, point2, point3]
        if rect_contains(rect, point1) and rect_contains(rect, point2) and rect_contains(rect, point3):
            indices = []
            for i in range(0, 3):
                for j in range(0, len(points)):
                    if abs(triple[i][1] - points[j][1]) < 1 and abs(triple[i][0] - points[j][0]) < 1:
                        indices.append(j)
            if len(indices) == 3:
                delaunay.append((indices[0], indices[1], indices[2]))
    return delaunay


def constrain_point(p, w, h) :
    return min(max(p[0],0), w - 1), min(max(p[1],0), h - 1)


def applyAffineTransform(src, srcTri, dstTri, size) :
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    return cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def warpTriangle(img1, img2, triangle1, triangle2) :
    rect1 = cv2.boundingRect(np.float32([triangle1]))
    rect2 = cv2.boundingRect(np.float32([triangle2]))

    t1Rect = [] 
    t2Rect = []
    t2RectInt = []
    for i in range(0, 3):
        t1Rect.append(((triangle1[i][0] - rect1[0]), (triangle1[i][1] - rect1[1])))
        t2Rect.append(((triangle2[i][0] - rect2[0]), (triangle2[i][1] - rect2[1])))
        t2RectInt.append(((triangle2[i][0] - rect2[0]),(triangle2[i][1] - rect2[1])))

    mask = np.zeros((rect2[3], rect2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    size = (rect2[2], rect2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] + img2Rect


def makeAverageImage(allPoints, images):
    global _height
    global _width

    eyecorner_dist = [(int(0.3 * _width), int(_height / 3)), (int(0.7 * _width), int(_height / 3))]
    imagesN = []
    pointsN = []
    boundary_points = np.array([(0,0), (_width/2, 0), (_width-1, 0), (_width-1, _height/2), (_width-1, _height-1 ), (_width/2, _height-1), (0, _height-1), (0, _height/2)])
    pointsAvg = np.array([(0,0)]* (len(allPoints[0]) + len(boundary_points)), np.float32())
    numImages = len(images)
    #################################################################
    # Applies similarity transform to each image to make each one 600 x 600 
    # centered on the face and transforms landmarks to correspond as well.
    # Also finds the average of the transformed landmarks.
    #################################################################
    for i in range(0, numImages):
        points1 = allPoints[i]
        # Corners of the eye in input image
        eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ]
        tform = similarity_transform(eyecornerSrc, eyecorner_dist)
        img = cv2.warpAffine(images[i], tform, (_width,_height))
        points2 = np.reshape(np.array(points1), (68,1,2))
        points = cv2.transform(points2, tform)
        points = np.float32(np.reshape(points, (68, 2)))
        points = np.append(points, boundary_points, axis=0)
        pointsAvg = pointsAvg + points / numImages
        pointsN.append(points)
        imagesN.append(img)
    
    # Delaunay triangulation
    rect = (0, 0, _width, _height)
    dt = calculate_triangles(rect, np.array(pointsAvg))
    output = np.zeros((_height, _width, 3), np.float32())
    for i in range(0, len(imagesN)):
        img = np.zeros((_height, _width, 3), np.float32())
        for j in range(0, len(dt)):
            tin = []
            tout = []
            for k in range(0, 3):
                tin.append(constrain_point(pointsN[i][dt[j][k]], _width, _height))
                tout.append(constrain_point(pointsAvg[dt[j][k]], _width, _height))
            warpTriangle(imagesN[i], img, tin, tout)
        output = output + img

    output = output/numImages
    output = cv2.convertScaleAbs(output, alpha=(255.0))
    return output

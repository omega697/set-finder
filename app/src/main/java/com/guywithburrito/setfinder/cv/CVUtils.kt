package com.guywithburrito.setfinder.cv

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Geometric utilities for OpenCV Mat operations.
 */
fun MatOfPoint2f.calculateIoU(other: MatOfPoint2f): Double {
    val intersectionPoly = MatOfPoint2f()
    // intersectConvexConvex returns the area of intersection
    val intersectArea = Imgproc.intersectConvexConvex(this, other, intersectionPoly)
    intersectionPoly.release()
    
    if (intersectArea <= 0) return 0.0
    
    val area1 = Imgproc.contourArea(this)
    val area2 = Imgproc.contourArea(other)
    val unionArea = area1 + area2 - intersectArea
    
    return if (unionArea > 0) intersectArea / unionArea else 0.0
}

/**
 * Calculates the centroid of the points in the Mat.
 */
fun MatOfPoint2f.getCenter(): Point {
    val points = this.toArray()
    if (points.isEmpty()) return Point(0.0, 0.0)
    return Point(points.sumOf { it.x } / points.size, points.sumOf { it.y } / points.size)
}

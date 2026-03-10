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

/**
 * Verifies if the quad contains white card stock by sampling 4 points inset from the corners.
 * Set cards are bright (>160 L) and neutral (<15.0 saturation).
 * [lab] must be an 8-bit 3-channel Mat in Lab color space.
 */
fun MatOfPoint2f.isWhiteCard(lab: Mat): Boolean {
    val pts = this.toArray()
    if (pts.size != 4) return false
    
    // Geometric center of the quad
    val cx = pts.sumOf { it.x } / 4.0
    val cy = pts.sumOf { it.y } / 4.0
    val center = Point(cx, cy)
    
    var totalL = 0.0
    var totalChrom = 0.0
    val sampleSize = 3
    
    // Sample 4 points 20% of the way from corner to center (the "white margin")
    for (p in pts) {
        val sx = (p.x * 0.8 + center.x * 0.2).toInt()
        val sy = (p.y * 0.8 + center.y * 0.2).toInt()
        
        val sampleRect = Rect(sx - sampleSize/2, sy - sampleSize/2, sampleSize, sampleSize)
        if (sampleRect.x < 0 || sampleRect.y < 0 || sampleRect.x + sampleRect.width > lab.cols() || sampleRect.y + sampleRect.height > lab.rows()) return false
        
        val roi = lab.submat(sampleRect)
        val mean = Core.mean(roi)
        roi.release()
        
        totalL += mean.`val`[0]
        val a = mean.`val`[1] - 128.0; val b = mean.`val`[2] - 128.0
        totalChrom += Math.sqrt(a * a + b * b)
    }
    
    val avgL = totalL / 4.0
    val avgChrom = totalChrom / 4.0
    
    return avgL > 160.0 && avgChrom < 15.0
}

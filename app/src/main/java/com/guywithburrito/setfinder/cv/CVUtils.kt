package com.guywithburrito.setfinder.cv

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Geometric utilities for OpenCV Mat operations.
 */
fun MatOfPoint2f.calculateIoU(other: MatOfPoint2f): Double {
    val intersectionPoly = MatOfPoint2f()
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
 * Verifies if the quad contains white card stock. 
 * Optimized for Tonemapped/SDR frames where white is truly bright (>180 L) 
 * and neutral (<10.0 saturation).
 * [lab] must be an 8-bit 3-channel Mat in Lab color space.
 */
fun MatOfPoint2f.isWhiteCard(lab: Mat): Boolean {
    val pts = this.toArray()
    if (pts.size != 4) return false
    
    val center = getCenter()
    
    var totalL = 0.0
    var totalChrom = 0.0
    val sampleSize = 3
    var validSamples = 0
    
    // Sample a 3x3 grid across the card to ensure we hit the white stock 
    // and not just a bunch of symbols.
    for (ix in 1..3) {
        for (iy in 1..3) {
            // Sample points at 25%, 50%, and 75% across both axes
            val fx = ix * 0.25
            val fy = iy * 0.25
            
            // Bilinear interpolation for the sample point
            val topX = pts[0].x * (1 - fx) + pts[1].x * fx
            val topY = pts[0].y * (1 - fx) + pts[1].y * fx
            val botX = pts[3].x * (1 - fx) + pts[2].x * fx
            val botY = pts[3].y * (1 - fx) + pts[2].y * fx
            
            val sx = (topX * (1 - fy) + botX * fy).toInt()
            val sy = (topY * (1 - fy) + botY * fy).toInt()
            
            val sampleRect = Rect(sx - sampleSize/2, sy - sampleSize/2, sampleSize, sampleSize)
            if (sampleRect.x < 0 || sampleRect.y < 0 || sampleRect.x + sampleRect.width > lab.cols() || sampleRect.y + sampleRect.height > lab.rows()) continue
            
            val roi = lab.submat(sampleRect)
            val mean = Core.mean(roi)
            roi.release()
            
            totalL += mean.`val`[0]
            val a = mean.`val`[1] - 128.0; val b = mean.`val`[2] - 128.0
            totalChrom += Math.sqrt(a * a + b * b)
            validSamples++
        }
    }
    
    if (validSamples == 0) return false
    
    val avgL = totalL / validSamples
    val avgChrom = totalChrom / validSamples
    
    // Tonemapped cards are VERY bright and VERY neutral.
    // Rocks and backgrounds usually have much higher chrominance or lower L.
    return avgL > 180.0 && avgChrom < 10.0
}

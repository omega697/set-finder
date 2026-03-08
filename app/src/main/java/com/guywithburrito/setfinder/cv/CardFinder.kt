package com.guywithburrito.setfinder.cv

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Robust CardFinder aligned with training data, but tuned for Android real-time.
 */
class CardFinder(private val settingsManager: com.guywithburrito.setfinder.tracking.SettingsManager? = null) {

    fun findCandidates(mat: Mat): List<MatOfPoint2f> {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGB2GRAY)
        
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        
        val thresh = Mat()
        val sensitivity = settingsManager?.sensitivity ?: 0.7f
        val blockSize = if (sensitivity > 0.85f) 7 else 11
        Imgproc.adaptiveThreshold(blurred, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, blockSize, 2.0)
        
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(thresh, contours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        
        val candidates = mutableListOf<MatOfPoint2f>()
        val frameArea = mat.width() * mat.height()
        val approx = MatOfPoint2f()
        
        for (cnt in contours) {
            val area = Imgproc.contourArea(cnt)
            // Allow cards to take up to 80% of the frame
            if (area < frameArea / 2000.0 || area > frameArea * 0.8) continue
            
            val peri = Imgproc.arcLength(MatOfPoint2f(*cnt.toArray()), true)
            // 0.02 is standard for quads that might be slightly rounded
            Imgproc.approxPolyDP(MatOfPoint2f(*cnt.toArray()), approx, 0.02 * peri, true)
            
            // Must be exactly 4 points and convex
            if (approx.total() == 4L && Imgproc.isContourConvex(MatOfPoint(*approx.toArray()))) {
                val rect = Imgproc.boundingRect(cnt)
                val ratio = rect.width.toDouble() / rect.height.toDouble()
                val invRatio = 1.0 / ratio
                // Cards are ~1.5 or ~0.66 ratio. Allow a wide range but block extreme skews.
                if (Math.max(ratio, invRatio) < 2.5) {
                    candidates.add(MatOfPoint2f(*approx.toArray()))
                }
            }
        }
        
        val filtered = filterDuplicatesIoU(candidates)
        Log.d("CardFinder", "Found ${filtered.size} raw candidates.")
        return filtered
    }

    fun findLikelyCards(mat: Mat): List<MatOfPoint2f> {
        return findCandidates(mat)
    }

    private fun filterDuplicatesIoU(candidates: List<MatOfPoint2f>): List<MatOfPoint2f> {
        if (candidates.isEmpty()) return emptyList()
        val sorted = candidates.sortedByDescending { Imgproc.contourArea(it) }
        val unique = mutableListOf<MatOfPoint2f>()
        for (cand in sorted) {
            val candRect = Imgproc.boundingRect(MatOfPoint(*cand.toArray()))
            var isDuplicate = false
            for (u in unique) {
                val uRect = Imgproc.boundingRect(MatOfPoint(*u.toArray()))
                val intersect = candRect.intersect(uRect)
                if (intersect != null) {
                    val intersectionArea = (intersect.width * intersect.height).toDouble()
                    val candArea = (candRect.width * candRect.height).toDouble()
                    val uArea = (uRect.width * uRect.height).toDouble()
                    
                    // If one is mostly inside another, it's likely a duplicate/nested contour
                    if (intersectionArea / candArea > 0.8 || intersectionArea / uArea > 0.8) {
                        isDuplicate = true; break
                    }
                    
                    val unionArea = candArea + uArea - intersectionArea
                    if (intersectionArea / unionArea > 0.6) { 
                        isDuplicate = true; break
                    }
                }
            }
            if (!isDuplicate) unique.add(cand)
        }
        return unique.take(20)
    }

    private fun Rect.intersect(other: Rect): Rect? {
        val x1 = Math.max(this.x, other.x)
        val y1 = Math.max(this.y, other.y)
        val x2 = Math.min(this.x + this.width, other.x + other.width)
        val y2 = Math.min(this.y + this.height, other.y + other.height)
        return if (x2 > x1 && y2 > y1) Rect(x1, y1, x2 - x1, y2 - y1) else null
    }
}

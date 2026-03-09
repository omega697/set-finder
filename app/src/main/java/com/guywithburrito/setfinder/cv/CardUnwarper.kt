package com.guywithburrito.setfinder.cv

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Aligned with chip_extractor.py logic for corner sorting and dimensions.
 */
open class CardUnwarper {
    companion object {
        const val TARGET_WIDTH = 144.0
        const val TARGET_HEIGHT = 224.0
    }

    fun unwarp(frame: Mat, corners: MatOfPoint2f): Mat {
        var rectified = rectify(corners.toArray())
        
        val p0 = rectified[0]; val p1 = rectified[1]; val p2 = rectified[2]; val p3 = rectified[3]
        val width = Math.sqrt(Math.pow(p1.x - p0.x, 2.0) + Math.pow(p1.y - p0.y, 2.0))
        val height = Math.sqrt(Math.pow(p3.x - p0.x, 2.0) + Math.pow(p3.y - p0.y, 2.0))
        
        if (width > height) {
            // Swap to make it portrait: bl, tl, tr, br
            rectified = arrayOf(rectified[3], rectified[0], rectified[1], rectified[2])
        }
        
        val src = MatOfPoint2f(*rectified)
        val dst = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(TARGET_WIDTH - 1, 0.0),
            Point(TARGET_WIDTH - 1, TARGET_HEIGHT - 1),
            Point(0.0, TARGET_HEIGHT - 1)
        )
        
        val result = Mat()
        val trans = Imgproc.getPerspectiveTransform(src, dst)
        Imgproc.warpPerspective(frame, result, trans, Size(TARGET_WIDTH, TARGET_HEIGHT))
        
        // Cleanup
        trans.release(); src.release(); dst.release()
        
        return result
    }

    private fun rectify(pts: Array<Point>): Array<Point> {
        val sorted = Array(4) { Point() }
        val sums = pts.map { it.x + it.y }
        val diffs = pts.map { it.y - it.x }
        
        sorted[0] = pts[sums.indexOf(sums.minOrNull())] // Top-left
        sorted[2] = pts[sums.indexOf(sums.maxOrNull())] // Bottom-right
        sorted[1] = pts[diffs.indexOf(diffs.minOrNull())] // Top-right
        sorted[3] = pts[diffs.indexOf(diffs.maxOrNull())] // Bottom-left
        
        return sorted
    }
}

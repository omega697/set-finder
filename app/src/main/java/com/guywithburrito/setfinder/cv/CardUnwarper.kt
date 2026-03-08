package com.guywithburrito.setfinder.cv

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Aligned with chip_extractor.py logic for corner sorting, dimensions, and white balance.
 */
class CardUnwarper {
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
        
        return applyWhiteBalance(result)
    }

    /**
     * LAB-based white balance identical to chip_extractor.py.
     * Centers 'a' and 'b' channels around 128.
     */
    private fun applyWhiteBalance(img: Mat): Mat {
        val lab = Mat()
        Imgproc.cvtColor(img, lab, Imgproc.COLOR_RGB2Lab)
        
        val channels = mutableListOf<Mat>()
        Core.split(lab, channels)
        
        val l = channels[0]
        val a = channels[1]
        val b = channels[2]
        
        val aMean = Core.mean(a).`val`[0]
        val bMean = Core.mean(b).`val`[0]
        
        Core.add(a, Scalar(128.0 - aMean), a)
        Core.add(b, Scalar(128.0 - bMean), b)
        
        val balanced = Mat()
        Core.merge(channels, balanced)
        
        val result = Mat()
        Imgproc.cvtColor(balanced, result, Imgproc.COLOR_Lab2RGB)
        return result
    }

    /**
     * Identical logic to Python rectify(pts):
     * np.diff(pts, axis=1) is (y - x).
     */
    private fun rectify(pts: Array<Point>): Array<Point> {
        val sorted = Array(4) { Point() }
        val sums = pts.map { it.x + it.y }
        val diffs = pts.map { it.y - it.x } // Python np.diff(pts, axis=1) is y-x
        
        sorted[0] = pts[sums.indexOf(sums.minOrNull())] // Top-left
        sorted[2] = pts[sums.indexOf(sums.maxOrNull())] // Bottom-right
        sorted[1] = pts[diffs.indexOf(diffs.minOrNull())] // Top-right (min y-x)
        sorted[3] = pts[diffs.indexOf(diffs.maxOrNull())] // Bottom-left (max y-x)
        
        return sorted
    }
}

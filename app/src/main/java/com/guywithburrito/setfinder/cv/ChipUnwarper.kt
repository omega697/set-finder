package com.guywithburrito.setfinder.cv

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Aligned with chip_extractor.py logic for corner sorting and dimensions.
 */
class ChipUnwarper {
    companion object {
        const val TARGET_WIDTH = 144.0
        const val TARGET_HEIGHT = 224.0
    }

    fun unwarp(frame: Mat, corners: MatOfPoint2f): Mat {
        val (p0, p1, p2, p3) = rectify(corners.toArray())
        val width = sqrt((p1.x - p0.x).pow(2.0) + (p1.y - p0.y).pow(2.0))
        val height = sqrt((p3.x - p0.x).pow(2.0) + (p3.y - p0.y).pow(2.0))


        val src =
            // Possibly swap to make it portrait: bl, tl, tr, br
            if (width > height) MatOfPoint2f(p3, p0, p1, p2)
            else MatOfPoint2f(p0, p1, p2, p3)

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

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
        val pts = corners.toArray()
        val (tl, tr, br, bl) = rectify(pts)
        
        // Calculate side lengths for aspect ratio check
        val widthA = sqrt((br.x - bl.x).pow(2.0) + (br.y - bl.y).pow(2.0))
        val widthB = sqrt((tr.x - tl.x).pow(2.0) + (tr.y - tl.y).pow(2.0))
        val heightA = sqrt((tr.x - br.x).pow(2.0) + (tr.y - br.y).pow(2.0))
        val heightB = sqrt((tl.x - bl.x).pow(2.0) + (tl.y - bl.y).pow(2.0))
        
        val avgWidth = (widthA + widthB) / 2.0
        val avgHeight = (heightA + heightB) / 2.0

        val src =
            if (avgWidth > avgHeight) MatOfPoint2f(bl, tl, tr, br)
            else MatOfPoint2f(tl, tr, br, bl)

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
        // Sort by Y coordinate
        val ySorted = pts.sortedBy { it.y }
        // Get top and bottom pairs
        val topPts = ySorted.take(2).sortedBy { it.x }
        val bottomPts = ySorted.drop(2).sortedBy { it.x }
        
        val tl = topPts[0]
        val tr = topPts[1]
        val bl = bottomPts[0]
        val br = bottomPts[1]
        
        return arrayOf(tl, tr, br, bl)
    }
}

package com.guywithburrito.setfinder.cv

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.atan2

/**
 * Robust card unwarping logic using Radial Sorting. 
 * Standardized with ml/tools/cv_library.py to ensure parity between 
 * training data and production app output.
 */
class ChipUnwarper {
    companion object {
        const val TARGET_WIDTH = 144.0
        const val TARGET_HEIGHT = 224.0
    }

    fun unwarp(frame: Mat, corners: MatOfPoint2f): Mat {
        val pts = corners.toArray()
        val rectified = rectify(pts)
        val tl = rectified[0]
        val tr = rectified[1]
        val br = rectified[2]
        val bl = rectified[3]
        
        val widthTop = sqrt((tr.x - tl.x).pow(2.0) + (tr.y - tl.y).pow(2.0))
        val widthBot = sqrt((br.x - bl.x).pow(2.0) + (br.y - bl.y).pow(2.0))
        val heightLeft = sqrt((tl.x - bl.x).pow(2.0) + (tl.y - bl.y).pow(2.0))
        val heightRight = sqrt((tr.x - br.x).pow(2.0) + (tr.y - br.y).pow(2.0))
        
        val avgWidth = (widthTop + widthBot) / 2.0
        val avgHeight = (heightLeft + heightRight) / 2.0

        // Mapping for source points to standard Portrait destination
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
        
        trans.release(); src.release(); dst.release()
        return result
    }

    /**
     * METHOD 2: RADIAL SORTING + TL ANCHOR
     * 1. Sorts points clockwise around their centroid.
     * 2. Identifies Top-Left as the point with the smallest x+y.
     * 3. Re-orders the list to start at Top-Left.
     */
    private fun rectify(pts: Array<Point>): Array<Point> {
        if (pts.size != 4) return pts

        // 1. Centroid
        val centerX = pts.sumOf { it.x } / 4.0
        val centerY = pts.sumOf { it.y } / 4.0
        
        // 2. Radial Sort (Clockwise)
        val clockwise = pts.sortedBy { atan2(it.y - centerY, it.x - centerX) }
        
        // 3. Anchor Top-Left (Min x+y)
        var tlIdx = 0
        var minSum = Double.MAX_VALUE
        for (i in clockwise.indices) {
            val sum = clockwise[i].x + clockwise[i].y
            if (sum < minSum) {
                minSum = sum
                tlIdx = i
            }
        }
        
        // 4. Array Shift
        return Array(4) { i -> clockwise[(tlIdx + i) % 4] }
    }
}

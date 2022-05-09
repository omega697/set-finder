package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer


class SetAnalyzer : ImageAnalysis.Analyzer {
    private val rects: MutableList<MatOfPoint> = mutableListOf()
    private val tempFrame1: Mat = Mat()
    var latestBitmap: Bitmap? = null
        private set

    override fun analyze(image: ImageProxy) {
        val frame: Mat = image.toMat()
        val contours: MutableList<MatOfPoint> = mutableListOf()
        val approx = MatOfPoint2f()

        rects.clear()
        // convert to HSV space, threshold, and find contours
        Imgproc.cvtColor(frame, tempFrame1, Imgproc.COLOR_RGB2HSV)
        Core.inRange(tempFrame1, CARD_HSV_LOWER_BOUND, CARD_HSV_UPPER_BOUND, tempFrame1)
        Imgproc.findContours(
            tempFrame1, contours, Mat(), Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        Log.d("SetAnalyzer", "Found ${contours.size} countours!")

        for (cnt in contours) {
            // approximate contours to get more regular shapes
            cnt.convertTo(approx, CvType.CV_32FC2)
            val sideErrorThresh: Double = SIDE_ERROR_SCALE * Imgproc.arcLength(approx, true)
            Imgproc.approxPolyDP(approx, approx, sideErrorThresh, true)
            // only take contours with 4 sides
            if (approx.total() != 4L) continue
            approx.convertTo(cnt, CvType.CV_32S)
            // apply area, convexity, and right-angle filters
            val cArea = Imgproc.contourArea(approx)
            if (cArea > MIN_RECT_AREA && cArea < MAX_RECT_AREA &&
                Imgproc.isContourConvex(cnt) && maxAngleCos(approx.toArray()) < MAX_CORNER_ANGLE_COS
            ) {
                rects.add(cnt)
            } else {
                Log.d("SetAnalyzer", "Area wrong: $cArea")
            }
        }

        if (rects.isNotEmpty()) Log.d("SetAnalyzer", "Found ${rects.size} rects!")

        image.close()
    }

    private fun ImageProxy.toMat(): Mat {
        val yBuffer: ByteBuffer = planes[0].buffer
        val uBuffer: ByteBuffer = planes[1].buffer
        val vBuffer: ByteBuffer = planes[2].buffer
        val ySize: Int = yBuffer.remaining()
        val uSize: Int = uBuffer.remaining()
        val vSize: Int = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuv = Mat(height + height / 2, width, CvType.CV_8UC1)
        yuv.put(0, 0, nv21)
        val mat = Mat()
        Imgproc.cvtColor(yuv, mat, Imgproc.COLOR_YUV2RGB_NV21, 3)
        return mat
    }

    companion object {
        const val SIDE_ERROR_SCALE = 0.1
        const val MAX_CORNER_ANGLE_COS = 0.3
        const val MIN_RECT_AREA = 1000
        const val MAX_RECT_AREA = 100000

        // Upper and lower bounds for card/shape thresholding.
        val CARD_HSV_LOWER_BOUND = Scalar(0.0, 0.0, 220.0)
        val CARD_HSV_UPPER_BOUND = Scalar(255.0, 70.0, 255.0)

        fun maxAngleCos(cnt: Array<Point>): Double {
            var maxCos = 0.0
            for (i in 2..4) {
                val cosine = Math.abs(angle(cnt[i % 4], cnt[i - 2], cnt[i - 1]))
                if (cosine > maxCos) {
                    maxCos = cosine
                }
            }
            return maxCos
        }

        fun angle(pt1: Point, pt2: Point, pt0: Point): Double {
            val dx1 = pt1.x - pt0.x
            val dy1 = pt1.y - pt0.y
            val dx2 = pt2.x - pt0.x
            val dy2 = pt2.y - pt0.y
            return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)
        }
    }
}
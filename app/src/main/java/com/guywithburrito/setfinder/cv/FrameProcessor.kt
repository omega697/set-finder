package com.guywithburrito.setfinder.cv

import org.opencv.core.*

/**
 * Interface for all OpenCV operations used by SetAnalyzer.
 * This allows for 100% JVM-only testing with mocks.
 */
interface FrameProcessor {
    fun createMat(): Mat
    fun createMatOfPoint2f(points: List<Point>): MatOfPoint2f
    fun resize(src: Mat, dst: Mat, size: Size, fx: Double, fy: Double, interpolation: Int)
    fun rotate(src: Mat, dst: Mat, rotationCode: Int)
    fun yuvToRgb(yuv: Mat, rgb: Mat)
    fun argmax(scores: FloatArray): Int
}

class OpenCVFrameProcessor : FrameProcessor {
    override fun createMat(): Mat = Mat()
    
    override fun createMatOfPoint2f(points: List<Point>): MatOfPoint2f {
        return MatOfPoint2f(*points.toTypedArray())
    }

    override fun resize(src: Mat, dst: Mat, size: Size, fx: Double, fy: Double, interpolation: Int) {
        org.opencv.imgproc.Imgproc.resize(src, dst, size, fx, fy, interpolation)
    }

    override fun rotate(src: Mat, dst: Mat, rotationCode: Int) {
        Core.rotate(src, dst, rotationCode)
    }

    override fun yuvToRgb(yuv: Mat, rgb: Mat) {
        org.opencv.imgproc.Imgproc.cvtColor(yuv, rgb, org.opencv.imgproc.Imgproc.COLOR_YUV2RGB_NV21)
    }

    override fun argmax(scores: FloatArray): Int {
        var bestIdx = 0; var maxVal = -1f
        for (i in scores.indices) { if (scores[i] > maxVal) { maxVal = scores[i]; bestIdx = i } }
        return bestIdx
    }
}

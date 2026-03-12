package com.guywithburrito.setfinder.cv

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Interface for color normalization to allow for mocking and easier testing.
 */
interface WhiteBalancer {
    /**
     * Balances an RGB image using LAB-based normalization.
     */
    fun balanceRGB(img: Mat): Mat
}

private fun calculateMedian(mat: Mat): Double {
    val temp = mat.reshape(1, 1)
    val sorted = Mat()
    Core.sort(temp, sorted, Core.SORT_EVERY_ROW or Core.SORT_ASCENDING)
    val medianIdx = sorted.cols() / 2
    val median = sorted.get(0, medianIdx)[0]
    sorted.release()
    temp.release()
    return median
}

/**
 * Standard implementation using OpenCV LAB conversion and channel shifting.
 * Aligned with refined chip_extractor.py logic.
 */
class OpenCVWhiteBalancer : WhiteBalancer {

    override fun balanceRGB(img: Mat): Mat {
        if (img.empty()) return img
        
        val lab = Mat()
        Imgproc.cvtColor(img, lab, Imgproc.COLOR_RGB2Lab)
        
        val channels = mutableListOf<Mat>()
        Core.split(lab, channels)
        
        // Use median to avoid being skewed by vibrant symbols
        val aMedian = calculateMedian(channels[1])
        val bMedian = calculateMedian(channels[2])
        
        // Shift to center at 128 (neutral in OpenCV 8-bit LAB)
        Core.add(channels[1], Scalar(128.0 - aMedian), channels[1])
        Core.add(channels[2], Scalar(128.0 - bMedian), channels[2])
        
        val balancedLab = Mat()
        Core.merge(channels, balancedLab)
        
        val result = Mat()
        Imgproc.cvtColor(balancedLab, result, Imgproc.COLOR_Lab2RGB)
        
        // Cleanup
        lab.release(); balancedLab.release()
        channels.forEach { it.release() }
        
        return result
    }
}

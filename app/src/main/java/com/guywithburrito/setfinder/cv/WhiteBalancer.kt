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

/**
 * Standard implementation using OpenCV LAB conversion and channel shifting.
 * Aligned with chip_extractor.py logic.
 */
class OpenCVWhiteBalancer : WhiteBalancer {

    override fun balanceRGB(img: Mat): Mat {
        if (img.empty()) return img
        
        val lab = Mat()
        Imgproc.cvtColor(img, lab, Imgproc.COLOR_RGB2Lab)
        
        val channels = mutableListOf<Mat>()
        Core.split(lab, channels)
        
        // In OpenCV 8-bit Lab:
        // channels[1] = a (0..255, 128 is neutral)
        // channels[2] = b (0..255, 128 is neutral)
        
        val aMean = Core.mean(channels[1]).`val`[0]
        val bMean = Core.mean(channels[2]).`val`[0]
        
        // Shift using Scalar addition to center at 128
        Core.add(channels[1], Scalar(128.0 - aMean), channels[1])
        Core.add(channels[2], Scalar(128.0 - bMean), channels[2])
        
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

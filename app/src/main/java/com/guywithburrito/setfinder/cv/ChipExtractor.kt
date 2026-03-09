package com.guywithburrito.setfinder.cv

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Stage 1: Extract chips from a full frame.
 * Responsible for unwarping and color normalization (White Balance).
 * Output chips have the "natural" card aspect ratio (144x224).
 */
class ChipExtractor(
    private val unwarper: CardUnwarper = CardUnwarper(),
    private val whiteBalancer: WhiteBalancer = OpenCVWhiteBalancer()
) {
    /**
     * Extracts a 144x224 white-balanced RGB chip from a frame and quad.
     */
    fun extract(frame: Mat, quad: MatOfPoint2f): Bitmap {
        // 1. Unwarp (144x224) - CardUnwarper targets this size
        val warped = unwarper.unwarp(frame, quad)
        
        // 2. White Balance
        val balanced = whiteBalancer.balanceRGB(warped)
        
        // 3. Convert to Bitmap
        val rgba = Mat()
        Imgproc.cvtColor(balanced, rgba, Imgproc.COLOR_RGB2RGBA)
        val bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        
        // Cleanup
        warped.release(); balanced.release(); rgba.release()
        
        return bmp
    }
}

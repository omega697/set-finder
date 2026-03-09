package com.guywithburrito.setfinder.cv

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class WhiteBalancerInstrumentationTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun balanceRGB_centersAandBat128() {
        val balancer = OpenCVWhiteBalancer()
        
        // Create a biased "blue-ish/green-ish" image
        // We'll use a 100x100 RGB image with specific bias
        val biased = Mat(100, 100, CvType.CV_8UC3, Scalar(100.0, 50.0, 150.0))
        
        val balanced = balancer.balanceRGB(biased)
        
        // Convert to Lab to verify centering
        val lab = Mat()
        Imgproc.cvtColor(balanced, lab, Imgproc.COLOR_RGB2Lab)
        val channels = mutableListOf<Mat>()
        Core.split(lab, channels)
        
        val aMean = Core.mean(channels[1]).`val`[0]
        val bMean = Core.mean(channels[2]).`val`[0]
        
        // Lab channel 1 (a) and 2 (b) should be exactly 128 in 8-bit OpenCV Lab space after our logic
        // We allow a small tolerance for rounding
        assertThat(aMean).isWithin(1.0).of(128.0)
        assertThat(bMean).isWithin(1.0).of(128.0)
        
        biased.release(); balanced.release(); lab.release()
        channels.forEach { it.release() }
    }
}

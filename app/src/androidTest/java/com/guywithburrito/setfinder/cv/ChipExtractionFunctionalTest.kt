package com.guywithburrito.setfinder.cv

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Functional verification of the Chip Extraction pipeline.
 * Tests both the low-level ChipUnwarper and the high-level ChipExtractor.
 */
@RunWith(AndroidJUnit4::class)
class ChipExtractionFunctionalTest {

    private lateinit var testMat: Mat
    private lateinit var testQuad: MatOfPoint2f

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
        
        // Setup common test asset and quad
        testMat = loadAsset("scenes/cards_12_3_sets.jpg")
        testQuad = MatOfPoint2f(
            Point(100.0, 100.0),
            Point(300.0, 110.0),
            Point(310.0, 450.0),
            Point(95.0, 440.0)
        )
    }

    @Test
    fun cardUnwarper_producesCorrectDimensions() {
        val unwarper = ChipUnwarper()
        val chip = unwarper.unwarp(testMat, testQuad)

        verifyDimensions(chip)
        verifyContent(chip)

        chip.release()
    }

    @Test
    fun chipExtractor_producesBalancedStandardizedChip() {
        val extractor = ChipExtractor()
        val chipBmp = extractor.extract(testMat, testQuad)

        val chipMat = Mat()
        Utils.bitmapToMat(chipBmp, chipMat)
        
        verifyDimensions(chipMat)
        verifyWhiteBalance(chipMat)

        chipMat.release()
    }

    private fun verifyDimensions(mat: Mat) {
        assertThat(mat.cols().toDouble()).isEqualTo(ChipUnwarper.TARGET_WIDTH)
        assertThat(mat.rows().toDouble()).isEqualTo(ChipUnwarper.TARGET_HEIGHT)
    }

    private fun verifyContent(mat: Mat) {
        val mean = Core.mean(mat)
        val brightness = mean.`val`.take(3).average()
        assertThat(brightness).isAtLeast(10.0) // Not black
    }

    private fun verifyWhiteBalance(mat: Mat) {
        val rgb = Mat()
        Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_RGBA2RGB)
        
        val lab = Mat()
        Imgproc.cvtColor(rgb, lab, Imgproc.COLOR_RGB2Lab)
        val channels = mutableListOf<Mat>()
        Core.split(lab, channels)
        
        val aMean = Core.mean(channels[1]).`val`[0]
        val bMean = Core.mean(channels[2]).`val`[0]
        
        // Center of Lab color space is 128
        assertThat(aMean).isWithin(2.0).of(128.0)
        assertThat(bMean).isWithin(2.0).of(128.0)

        rgb.release(); lab.release()
        channels.forEach { it.release() }
    }

    private fun loadAsset(path: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val inputStream = testContext.assets.open(path)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }
}

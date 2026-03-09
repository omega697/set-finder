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
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class ChipExtractionInstrumentationTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun extract_producesBalancedStandardizedChip() {
        val extractor = ChipExtractor()
        val context = InstrumentationRegistry.getInstrumentation().context
        
        // 1. Load full frame (RGB)
        val inputStream = context.assets.open("scenes/cards_12_3_sets.jpg")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val frameRGB = Mat()
        Utils.bitmapToMat(bitmap, frameRGB)
        Imgproc.cvtColor(frameRGB, frameRGB, Imgproc.COLOR_RGBA2RGB)

        // 2. Define a known quad (likely a card)
        val quad = MatOfPoint2f(
            Point(100.0, 100.0),
            Point(300.0, 110.0),
            Point(310.0, 450.0),
            Point(95.0, 440.0)
        )

        // 3. Extract
        val chipBmp = extractor.extract(frameRGB, quad)

        // 4. Verify Dimensions
        assertThat(chipBmp.width).isEqualTo(144)
        assertThat(chipBmp.height).isEqualTo(224)
        
        // 5. Verify Balance
        val chipMat = Mat()
        Utils.bitmapToMat(chipBmp, chipMat)
        val chipRGB = Mat()
        Imgproc.cvtColor(chipMat, chipRGB, Imgproc.COLOR_RGBA2RGB)
        
        val lab = Mat()
        Imgproc.cvtColor(chipRGB, lab, Imgproc.COLOR_RGB2Lab)
        val channels = mutableListOf<Mat>()
        Core.split(lab, channels)
        
        val aMean = Core.mean(channels[1]).`val`[0]
        val bMean = Core.mean(channels[2]).`val`[0]
        
        // Should be centered at 128
        assertThat(aMean).isWithin(2.0).of(128.0)
        assertThat(bMean).isWithin(2.0).of(128.0)

        frameRGB.release(); quad.release(); chipMat.release(); chipRGB.release(); lab.release()
        channels.forEach { it.release() }
    }
}

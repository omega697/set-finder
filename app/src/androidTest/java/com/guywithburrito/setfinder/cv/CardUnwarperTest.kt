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

@RunWith(AndroidJUnit4::class)
class CardUnwarperTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun unwarp_producesCorrectDimensions() {
        val unwarper = CardUnwarper()
        val context = InstrumentationRegistry.getInstrumentation().context
        
        // 1. Load sample frame
        val inputStream = context.assets.open("scenes/cards_12_3_sets.jpg")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val frameRGB = Mat()
        Utils.bitmapToMat(bitmap, frameRGB)
        Imgproc.cvtColor(frameRGB, frameRGB, Imgproc.COLOR_RGBA2RGB)

        // 2. Define a known quad (approximate)
        val quad = MatOfPoint2f(
            Point(100.0, 100.0),
            Point(300.0, 110.0),
            Point(310.0, 450.0),
            Point(95.0, 440.0)
        )

        // 3. Unwarp
        val chip = unwarper.unwarp(frameRGB, quad)

        // 4. Assertions
        assertThat(chip.cols().toDouble()).isEqualTo(CardUnwarper.TARGET_WIDTH)
        assertThat(chip.rows().toDouble()).isEqualTo(CardUnwarper.TARGET_HEIGHT)
        
        val mean = Core.mean(chip)
        val brightness = mean.`val`.take(3).average()
        assertThat(brightness).isAtLeast(10.0) // Not black

        frameRGB.release(); quad.release(); chip.release()
    }
}

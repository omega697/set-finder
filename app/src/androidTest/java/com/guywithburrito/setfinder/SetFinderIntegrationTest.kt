package com.guywithburrito.setfinder

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlinx.coroutines.MainScope

@RunWith(AndroidJUnit4::class)
class SetFinderIntegrationTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun analyze_findsThreeSetsInSampleImage() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val analyzer = SetAnalyzer(appContext, MainScope())
        
        // Load the 12-card test image
        val mat = loadAsset("cards_12_3_sets.jpg")
        
        // Run analysis (this will use TFLiteCardIdentifier internally)
        analyzer.analyzeMat(mat)
        
        // Check results
        // 1. Should have identified 12 cards (or close to it)
        assertThat(analyzer.detectedRects.size).isAtLeast(10)
        
        // 2. Should have found exactly 3 sets
        assertThat(analyzer.foundSets.size).isEqualTo(3)
        
        // 3. Check the colors are distinct
        val setColors = analyzer.foundSets.map { it.hashCode() }
        assertThat(setColors.distinct().size).isEqualTo(3)
    }

    private fun loadAsset(assetName: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val inputStream = context.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }
}

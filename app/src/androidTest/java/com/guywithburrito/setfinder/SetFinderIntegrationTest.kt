package com.guywithburrito.setfinder

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import com.guywithburrito.setfinder.cv.*
import com.guywithburrito.setfinder.ml.*
import com.guywithburrito.setfinder.tracking.SettingsManager
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class SetFinderIntegrationTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun analyze_findsThreeSetsInSampleImage() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val settingsManager = SettingsManager(appContext)
        val detector = SetDetector(
            CardFinder(settingsManager),
            CardUnwarper(),
            TFLiteCardIdentifier(
                TFLiteCardFilterModel(appContext), 
                TFLiteExpertModel(appContext), 
                CardModelMapper.V12, 
                OpenCVWhiteBalancer()
            )
        )
        
        // Load the 12-card test image (Ground Truth: has 3+ sets)
        val mat = loadAsset("cards_12_3_sets.jpg")
        
        // One-shot modular pipeline (Sync)
        val sets = detector.detectSets(mat)
        
        android.util.Log.d("IntegrationTest", "Found ${sets.size} sets in one-shot analysis.")
        
        // 1. Should have found at least 3 sets
        assertThat(sets.size).isAtLeast(3)
        
        mat.release()
        detector.close()
    }

    private fun loadAsset(assetName: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val inputStream = testContext.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }
}

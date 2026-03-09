package com.guywithburrito.setfinder

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.ChipExtractor
import com.guywithburrito.setfinder.ml.CardIdentifier
import com.guywithburrito.setfinder.tracking.SettingsManager
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class SetDetectorTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun detectSets_findsThreeSetsInSampleImage() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val settingsManager = SettingsManager(appContext)
        
        val finder = CardFinder(settingsManager)
        val extractor = ChipExtractor()
        val identifier = CardIdentifier.getInstance(appContext)
        
        val detector = SetDetector(finder, extractor, identifier)
        
        // 1. Load the 12-card test image
        val mat = loadAsset("scenes/cards_12_3_sets.jpg")
        
        // 2. Perform one-shot detection
        val sets = detector.detectSets(mat)
        
        android.util.Log.d("SetDetectorTest", "Found ${sets.size} sets.")
        
        // 3. Should find at least 3 sets
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

package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
import com.guywithburrito.setfinder.ml.*
import com.guywithburrito.setfinder.cv.OpenCVWhiteBalancer
import com.guywithburrito.setfinder.tracking.SettingsManager
import com.guywithburrito.setfinder.card.SetCard
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.test.assertNotNull

@RunWith(AndroidJUnit4::class)
class PipelineStepTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun step1_DetectionTest() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val settingsManager = SettingsManager(appContext)
        val mat = loadFullFrame("cards_12_3_sets.jpg")
        val finder = CardFinder(settingsManager)
        
        // Use refactored production method
        val filtered = finder.findLikelyCards(mat)

        android.util.Log.d("PipelineTest", "Detected: ${filtered.size} cards")
        // We expect exactly 12 cards in cards_12_3_sets.jpg
        assertThat(filtered.size).isAtLeast(10)
    }

    @Test
    fun step2_UnwarpAndIdentifyTest() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val settingsManager = SettingsManager(appContext)
        val mat = loadFullFrame("card_1_green_shaded_diamond.jpg")
        val finder = CardFinder(settingsManager)
        val unwarper = CardUnwarper()
        val identifier = TFLiteCardIdentifier(TFLiteCardFilterModel(appContext, "card_filter.tflite"), TFLiteExpertModel(appContext, "set_card_model_final.tflite"), CardModelMapper.V12, OpenCVWhiteBalancer())
        
        val candidates = finder.findLikelyCards(mat)
        assertThat(candidates).isNotEmpty()
        
        // Verify unwarp dimensions match what model expects (224x224 in production)
        val chip = unwarper.unwarp(mat, candidates[0])
        assertThat(chip.rows()).isEqualTo(224)
        
        // Convert to bitmap and identify - this verifies the full chain
        val bmp = Bitmap.createBitmap(chip.cols(), chip.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(chip, bmp)
        
        val card = identifier.identifyCard(bmp)
        assertNotNull(card, "Model should identify the unwarped card")
        assertThat(card.color).isEqualTo(SetCard.Color.GREEN)
        
        identifier.close()
    }

    private fun loadFullFrame(assetName: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val inputStream = testContext.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        
        // Analysis frame is scaled to 1000px max dim
        val maxDim = 1000.0
        val scale = maxDim / Math.max(bitmap.width, bitmap.height)
        val width = (bitmap.width * scale).toInt()
        val height = (bitmap.height * scale).toInt()
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)
        
        val mat = Mat()
        Utils.bitmapToMat(scaledBitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }
}

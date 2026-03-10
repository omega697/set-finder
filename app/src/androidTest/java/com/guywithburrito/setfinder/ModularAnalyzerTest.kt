package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.OpenCVQuadFinder
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
import kotlin.test.assertNotNull

/**
 * This test evaluates the integrated performance of the card detection and identification 
 * components on full-resolution scene assets. It ensures that the candidate finder and 
 * attribute expert work together correctly to identify specific cards within real-world 
 * images, serving as a functional validation of the combined vision and ML pipeline.
 */
@RunWith(AndroidJUnit4::class)
class ModularAnalyzerTest {

    private lateinit var detector: SetDetector

    @Before
    fun setUp() {
        if (!OpenCVLoader.initDebug()) {
            throw RuntimeException("OpenCV initialization failed!")
        }
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val settingsManager = SettingsManager(context)
        val finder = OpenCVQuadFinder()
        val extractor = ChipExtractor()
        val identifier = CardIdentifier.getInstance(context)
        
        detector = SetDetector(finder, extractor, identifier)
    }

    @Test
    fun scene_GreenShadedDiamond_IdentifyFull() {
        val mat = loadAsset("scenes/scene_two_green_shaded_diamond.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.GREEN)
        assertThat(card.count).isEqualTo(SetCard.Count.TWO)
        assertThat(card.shape).isEqualTo(SetCard.Shape.DIAMOND)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.SHADED)
    }

    @Test
    fun scene_RedSolidOval_IdentifyFull() {
        val mat = loadAsset("scenes/card_3_red_solid_oval.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.RED)
        assertThat(card.count).isEqualTo(SetCard.Count.THREE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.OVAL)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.SOLID)
    }

    @Test
    fun scene_PurpleEmptyOval_IdentifyFull() {
        val mat = loadAsset("scenes/card_1_purple_empty_oval.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.PURPLE)
        assertThat(card.count).isEqualTo(SetCard.Count.ONE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.OVAL)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.EMPTY)
    }

    @Test
    fun scene_RedEmptySquiggle_IdentifyFull() {
        val mat = loadAsset("scenes/card_3_red_empty_squiggle.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.RED)
        assertThat(card.count).isEqualTo(SetCard.Count.THREE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.SQUIGGLE)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.EMPTY)
    }

    @Test
    fun scene_RedShadedDiamond_IdentifyFull() {
        val mat = loadAsset("scenes/card_1_red_shaded_diamond.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.RED)
        assertThat(card.count).isEqualTo(SetCard.Count.ONE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.DIAMOND)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.SHADED)
    }

    @Test
    fun scene_Kindle_ShouldReturnNoCards() {
        val mat = loadAsset("scenes/desk_no_cards.jpg")
        val detected = detector.detectCards(mat)
        assertThat(detected).isEmpty()
    }

    private fun identifyFirstCard(mat: Mat): SetCard? {
        val detected = detector.detectCards(mat)
        return detected.firstOrNull()?.card
    }

    private fun loadAsset(assetName: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val inputStream = context.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        
        // Scale to standard analysis size (1000px max dim)
        val maxDim = 1000.0
        val scale = maxDim / Math.max(bitmap.width, bitmap.height)
        val width = (bitmap.width * scale).toInt()
        val height = (bitmap.height * scale).toInt()
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)
        
        val mat = Mat()
        Utils.bitmapToMat(scaledBitmap, mat)
        
        val rgb = Mat()
        Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_RGBA2RGB)
        mat.release()
        return rgb
    }
}

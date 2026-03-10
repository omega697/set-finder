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
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import kotlin.test.assertNotNull

/**
 * Functional validation of the integrated vision and ML pipeline.
 * Tests detection, extraction, and identification on real-world scene assets.
 */
@RunWith(AndroidJUnit4::class)
class ModularAnalyzerTest {

    private lateinit var finder: OpenCVQuadFinder
    private lateinit var extractor: ChipExtractor
    private lateinit var identifier: CardIdentifier

    @Before
    fun setUp() {
        if (!OpenCVLoader.initDebug()) {
            throw RuntimeException("OpenCV initialization failed!")
        }
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        finder = OpenCVQuadFinder()
        extractor = ChipExtractor()
        identifier = CardIdentifier.getInstance(context)
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
    fun scene_Kindle_ShouldReturnNoCards() {
        val mat = loadAsset("scenes/desk_no_cards.jpg")
        val candidates = finder.findCandidates(mat)
        assertThat(candidates).isEmpty()
    }

    private fun identifyFirstCard(mat: Mat): SetCard? {
        val candidates = finder.findCandidates(mat)
        val first = candidates.firstOrNull() ?: return null
        val chip = extractor.extract(mat, first)
        return identifier.identifyCard(chip)
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

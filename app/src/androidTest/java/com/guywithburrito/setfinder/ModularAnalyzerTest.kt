package com.guywithburrito.setfinder

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import com.guywithburrito.setfinder.card.SetCard
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
import kotlin.test.assertNotNull

@RunWith(AndroidJUnit4::class)
class ModularAnalyzerTest {

    private lateinit var finder: CardFinder
    private val extractor = ChipExtractor()
    private lateinit var identifier: CardIdentifier

    @Before
    fun setUp() {
        if (!OpenCVLoader.initDebug()) {
            throw RuntimeException("OpenCV initialization failed!")
        }
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val settingsManager = SettingsManager(context)
        finder = CardFinder(settingsManager)
        identifier = CardIdentifier.getInstance(context)
    }

    @Test
    fun stage4_GreenShadedDiamond_IdentifyFull() {
        val mat = loadAsset("card_1_green_shaded_diamond.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.GREEN)
        assertThat(card.count).isEqualTo(SetCard.Count.ONE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.DIAMOND)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.SHADED)
    }

    @Test
    fun stage4_RedSolidOval_IdentifyFull() {
        val mat = loadAsset("card_3_red_solid_oval.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.RED)
        assertThat(card.count).isEqualTo(SetCard.Count.THREE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.OVAL)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.SOLID)
    }

    @Test
    fun stage4_PurpleEmptyOval_IdentifyFull() {
        val mat = loadAsset("card_1_purple_empty_oval.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.PURPLE)
        assertThat(card.count).isEqualTo(SetCard.Count.ONE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.OVAL)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.EMPTY)
    }

    @Test
    fun stage4_RedEmptySquiggle_IdentifyFull() {
        val mat = loadAsset("card_3_red_empty_squiggle.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.RED)
        assertThat(card.count).isEqualTo(SetCard.Count.THREE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.SQUIGGLE)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.EMPTY)
    }

    @Test
    fun stage4_RedShadedDiamond_IdentifyFull() {
        val mat = loadAsset("card_1_red_shaded_diamond.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.RED)
        assertThat(card.count).isEqualTo(SetCard.Count.ONE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.DIAMOND)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.SHADED)
    }

    @Test
    fun stage4_Kindle_ShouldReturnNull() {
        val mat = loadAsset("desk_no_cards.jpg")
        val candidates = finder.findCandidates(mat)
        val card = candidates.mapNotNull { quad -> 
            val chip = extractor.extract(mat, quad)
            identifier.identifyCard(chip) 
        }.firstOrNull()
        
        assertThat(card).isNull()
    }

    private fun identifyFirstCard(mat: Mat): SetCard? {
        val candidates = finder.findCandidates(mat)
        if (candidates.isEmpty()) return null
        
        return candidates.mapNotNull { quad ->
            val chip = extractor.extract(mat, quad)
            identifier.identifyCard(chip)
        }.firstOrNull()
    }

    private fun loadAsset(assetName: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val inputStream = context.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        
        val rgb = Mat()
        Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_RGBA2RGB)
        mat.release()
        return rgb
    }
}

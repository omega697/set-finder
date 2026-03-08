package com.guywithburrito.setfinder

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import com.guywithburrito.setfinder.card.*
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
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

    private val finder = CardFinder()
    private val unwarper = CardUnwarper()
    private lateinit var identifier: TFLiteCardIdentifier

    @Before
    fun setUp() {
        if (!OpenCVLoader.initDebug()) {
            throw RuntimeException("OpenCV initialization failed!")
        }
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        identifier = TFLiteCardIdentifier(context)
    }

    @Test
    fun stage4_GreenStripedDiamond_IdentifyFull() {
        val mat = loadAsset("card_1_green_striped_diamond.jpg")
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
    fun stage4_PurpleOpenOval_IdentifyFull() {
        val mat = loadAsset("card_1_purple_open_oval.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.PURPLE)
        assertThat(card.count).isEqualTo(SetCard.Count.ONE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.OVAL)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.EMPTY)
    }

    @Test
    fun stage4_RedOpenSquiggle_IdentifyFull() {
        val mat = loadAsset("card_3_red_open_squiggle.jpg")
        val card = identifyFirstCard(mat)
        
        assertNotNull(card)
        assertThat(card.color).isEqualTo(SetCard.Color.RED)
        assertThat(card.count).isEqualTo(SetCard.Count.THREE)
        assertThat(card.shape).isEqualTo(SetCard.Shape.SQUIGGLE)
        assertThat(card.pattern).isEqualTo(SetCard.Pattern.EMPTY)
    }

    @Test
    fun stage4_RedStripedDiamond_IdentifyFull() {
        val mat = loadAsset("card_1_red_striped_diamond.jpg")
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
            val warped = unwarper.unwarp(mat, quad)
            val bmp = android.graphics.Bitmap.createBitmap(warped.cols(), warped.rows(), android.graphics.Bitmap.Config.ARGB_8888)
            org.opencv.android.Utils.matToBitmap(warped, bmp)
            identifier.identifyCard(bmp) 
        }.firstOrNull()
        
        assertThat(card).isNull()
    }

    private fun identifyFirstCard(mat: Mat): SetCard? {
        val candidates = finder.findCandidates(mat)
        if (candidates.isEmpty()) return null
        
        // Return the first successfully identified card
        return candidates.mapNotNull { quad ->
            val warped = unwarper.unwarp(mat, quad)
            val bmp = android.graphics.Bitmap.createBitmap(warped.cols(), warped.rows(), android.graphics.Bitmap.Config.ARGB_8888)
            org.opencv.android.Utils.matToBitmap(warped, bmp)
            identifier.identifyCard(bmp)
        }.firstOrNull()
    }

    private fun loadAsset(assetName: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val inputStream = context.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        
        val maxDim = 1000.0
        val scale = maxDim / Math.max(bitmap.width, bitmap.height)
        val width = (bitmap.width * scale).toInt()
        val height = (bitmap.height * scale).toInt()
        val scaledBitmap = android.graphics.Bitmap.createScaledBitmap(bitmap, width, height, true)
        
        val mat = Mat()
        Utils.bitmapToMat(scaledBitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }
}

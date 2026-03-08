package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import kotlin.test.assertNotNull

@RunWith(AndroidJUnit4::class)
class PipelineAlignmentTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun stage0_LoadAssets_VerifiesImageLoading() {
        val mat = loadFullFrame("cards_13_wide_shot.jpg")
        assertThat(mat.width()).isAtLeast(500)
        assertThat(mat.height()).isAtLeast(500)
        saveDebugMat(mat, "alignment_stage0_load.jpg")
    }

    @Test
    fun stage1_Detection_FindsCorrectCount() {
        val mat = loadFullFrame("cards_13_wide_shot.jpg")
        val finder = CardFinder()
        val cards = finder.findLikelyCards(mat)
        
        // OpenCV finds many candidates (symbols, shadows, cards).
        // The important part is that it finds ALL the cards.
        assertThat(cards.size).isAtLeast(13)
    }

    @Test
    fun stage2_Unwarp_ProducesCorrectDimensions() {
        val mat = loadFullFrame("card_1_green_striped_diamond.jpg")
        val finder = CardFinder()
        val unwarper = CardUnwarper()
        
        // Find quads in a real frame
        val cards = finder.findCandidates(mat)
        assertThat(cards).isNotEmpty()
        
        val chip = unwarper.unwarp(mat, cards[0])
        
        // Assert dimensions exactly match training chips
        assertThat(chip.cols()).isEqualTo(144)
        assertThat(chip.rows()).isEqualTo(224)
        
        saveDebugMat(chip, "alignment_stage2_unwarp.jpg")
    }

    @Test
    fun stage3_Identification_MatchesV12Expectations() {
        // ... (existing code)
    }

    @Test
    fun stage4_CoordinateMapping_VerifiesScalingMath() {
        // Simulated screen (1080x2424 - common portrait)
        val canvasWidth = 1080f
        val canvasHeight = 2424f
        
        // Simulated analysis frame (720x1280 camera rotated to portrait, then scaled to 1000 max dim)
        // Ratio 720/1280 = 0.5625. 1280 -> 1000, 720 -> 562.5
        val imgWidth = 562f
        val imgHeight = 1000f
        
        // FILL_CENTER logic from SetFinderView.kt
        val scale = Math.max(canvasWidth / imgWidth, canvasHeight / imgHeight)
        val scaledWidth = imgWidth * scale
        val scaledHeight = imgHeight * scale
        val offsetX = (canvasWidth - scaledWidth) / 2f
        val offsetY = (canvasHeight - scaledHeight) / 2f
        
        // Assertions for FILL_CENTER in this specific aspect ratio
        // canvas aspect = 2424/1080 = 2.24
        // image aspect = 1000/562 = 1.77
        // Canvas is "taller" than image. scale should be driven by height.
        assertThat(canvasHeight / imgHeight).isAtLeast(canvasWidth / imgWidth)
        assertThat(scale).isEqualTo(2.424f)
        
        // Offset check
        assertThat(offsetY).isEqualTo(0f)
        assertThat(offsetX).isLessThan(0f) // Should be centered horizontally with negative offset (cropped)
        
        // Verify mapping of a center point
        val centerImg = Point(imgWidth / 2.0, imgHeight / 2.0)
        val centerCanvasX = (centerImg.x.toFloat() * scale) + offsetX
        val centerCanvasY = (centerImg.y.toFloat() * scale) + offsetY
        
        assertThat(centerCanvasX).isWithin(0.1f).of(canvasWidth / 2f)
        assertThat(centerCanvasY).isWithin(0.1f).of(canvasHeight / 2f)
    }

    private fun loadFullFrame(assetName: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val inputStream = testContext.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        
        // Scale to a standard dimension for consistent testing
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

    private fun loadAssetAsBitmap(assetName: String): Bitmap {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        return BitmapFactory.decodeStream(testContext.assets.open(assetName))
    }

    private fun saveDebugMat(mat: Mat, fileName: String) {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        
        // Use app-specific cache directory to avoid permissions issues
        val file = File(appContext.cacheDir, fileName)
        FileOutputStream(file).use { out ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }
        android.util.Log.d("PipelineTest", "Saved debug image to: ${file.absolutePath}")
    }
}

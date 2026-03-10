package com.guywithburrito.setfinder

import android.graphics.Bitmap
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
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class PipelineStepTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun stage4_EndToEnd_IdentifiesSpecificCard() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context
        
        // Setup Modular Components
        val extractor = ChipExtractor()
        val identifier = CardIdentifier.getInstance(appContext)
        
        // Use renamed asset in scenes folder
        val mat = loadFullFrame("scenes/scene_two_green_shaded_diamond.jpg")
        val scale = 1000.0 / Math.max(mat.cols().toDouble(), mat.rows().toDouble())
        val small = Mat()
        Imgproc.resize(mat, small, Size(), scale, scale, Imgproc.INTER_AREA)

        val finder = CardFinder()
        val candidates = finder.findCandidates(small)
        assertThat(candidates).isNotEmpty()

        val results = candidates.mapNotNull { quad ->
            val corners = quad.toArray().map { p -> Point(p.x / scale, p.y / scale) }
            val fullResQuad = MatOfPoint2f(*corners.toTypedArray())
            val chip = extractor.extract(mat, fullResQuad)
            identifier.identifyCard(chip)
        }

        assertThat(results).isNotEmpty()
        // Ensure color is correctly identified (GREEN)
        assertThat(results[0].color).isEqualTo(SetCard.Color.GREEN)
        
        identifier.close()
        mat.release(); small.release()
    }

    private fun loadFullFrame(assetName: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val inputStream = testContext.assets.open(assetName)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }
}

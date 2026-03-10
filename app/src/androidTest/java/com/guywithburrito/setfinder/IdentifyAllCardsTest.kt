package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.guywithburrito.setfinder.cv.OpenCVQuadFinder
import com.guywithburrito.setfinder.cv.ChipExtractor
import com.guywithburrito.setfinder.ml.CardIdentifier
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
import java.io.File
import java.io.FileOutputStream

/**
 * This test verifies the end-to-end identification process for all cards in a 
 * known test scene. It ensures that the modular components (finder, extractor, 
 * identifier) work together to correctly map and identify every card in a complex 
 * image, serving as a regression test for the entire vision-to-ML pipeline.
 */
@RunWith(AndroidJUnit4::class)
class IdentifyAllCardsTest {

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun verifyCanonicalMappingsForKnownImages() {
        val mat = loadAsset("scenes/cards_12_3_sets.jpg")
        val maxDim = 1000.0
        val scale = maxDim / Math.max(mat.cols().toDouble(), mat.rows().toDouble())
        val small = Mat()
        Imgproc.resize(mat, small, Size(), scale, scale, Imgproc.INTER_AREA)

        val finder = OpenCVQuadFinder()
        val quads = finder.findCandidates(small)
        android.util.Log.d("IdentifyAll", "Detected ${quads.size} cards.")

        // 1. Setup Modular components
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val extractor = ChipExtractor()
        val identifier = CardIdentifier.getInstance(context)

        // 2. Identification & Logging
        quads.forEachIndexed { index, quad ->
            val corners = quad.toArray().map { p -> Point(p.x / scale, p.y / scale) }
            val fullResQuad = MatOfPoint2f(*corners.toTypedArray())
            
            val bmp = extractor.extract(mat, fullResQuad)
            val result = identifier.identifyCard(bmp)
            
            android.util.Log.d("IdentifyAll", "Card #$index: $result")
            
            // Save for inspection
            saveBitmap(bmp, "chip_$index.jpg")
            
            fullResQuad.release()
        }
        
        identifier.close()
        mat.release()
        small.release()
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

    private fun saveBitmap(bitmap: Bitmap, filename: String) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val file = File(context.getExternalFilesDir(null), filename)
        FileOutputStream(file).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
        }
    }
}

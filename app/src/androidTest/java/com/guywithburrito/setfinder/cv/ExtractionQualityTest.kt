package com.guywithburrito.setfinder.cv

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.services.storage.TestStorage
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
import java.io.IOException

/**
 * This test serves as a debug utility for evaluating chip extraction quality across 
 * all test scenes. It batch-processes every scene asset, detects card candidates, 
 * and saves the resulting chips to device storage for visual inspection and 
 * ground-truth creation.
 */
@RunWith(AndroidJUnit4::class)
class ExtractionQualityTest {

    private val finder = OpenCVQuadFinder()
    private val extractor = ChipExtractor()

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun debug_ExtractAllChipsFromAllScenes() {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val scenes = testContext.assets.list("scenes") ?: emptyArray()
        
        val testStorage = TestStorage()

        android.util.Log.i("ExtractionQuality", "Extracting chips via TestStorage (JPEG)...")

        scenes.forEach { sceneName ->
            if (!sceneName.endsWith(".jpg")) return@forEach
            
            val mat = loadAsset("scenes/$sceneName")
            val maxDim = 1000.0
            val scale = maxDim / Math.max(mat.cols().toDouble(), mat.rows().toDouble())
            val small = Mat()
            Imgproc.resize(mat, small, Size(), scale, scale, Imgproc.INTER_AREA)

            val quads = finder.findCandidates(small)
            android.util.Log.i("ExtractionQuality", "Scene $sceneName: Found ${quads.size} quads.")

            quads.forEachIndexed { index, quad ->
                val corners = quad.toArray().map { p -> Point(p.x / scale, p.y / scale) }
                val fullResQuad = MatOfPoint2f(*corners.toTypedArray())
                
                try {
                    val chip = extractor.extract(mat, fullResQuad)
                    val filename = "${sceneName.removeSuffix(".jpg")}_chip_$index.jpg"
                    
                    testStorage.openOutputFile(filename).use { outputStream ->
                        val success = chip.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                        if (!success) {
                            throw IOException("Failed to compress chip $index from $sceneName to JPEG")
                        }
                    }
                } catch (e: Exception) {
                    android.util.Log.e("ExtractionQuality", "Failed to extract chip $index from $sceneName", e)
                } finally {
                    fullResQuad.release()
                }
            }
            
            mat.release()
            small.release()
        }
    }

    private fun loadAsset(path: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val inputStream = testContext.assets.open(path)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }
}

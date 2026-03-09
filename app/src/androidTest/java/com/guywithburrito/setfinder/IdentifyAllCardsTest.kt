package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
import com.guywithburrito.setfinder.ml.*
import com.guywithburrito.setfinder.cv.OpenCVWhiteBalancer
import com.guywithburrito.setfinder.tracking.SettingsManager
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.MainScope

@RunWith(AndroidJUnit4::class)
class IdentifyAllCardsTest {

    private lateinit var analyzer: SetAnalyzer

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val settingsManager = SettingsManager(appContext)
        analyzer = SetAnalyzer(appContext, MainScope(), settingsManager)
    }

    @Test
    fun logAllIdentifiedCards() {
        val mat = loadAsset("cards_12_3_sets.jpg")
        
        // 1. Detection
        val maxDim = 1000.0
        val scale = maxDim / Math.max(mat.cols().toDouble(), mat.rows().toDouble())
        val small = Mat()
        Imgproc.resize(mat, small, Size(), scale, scale, Imgproc.INTER_AREA)
        
        // Use the internal finder via reflection or just duplicate the logic for a clean test
        val finder = CardFinder(SettingsManager(InstrumentationRegistry.getInstrumentation().targetContext))
        val quads = finder.findLikelyCards(small)
        
        android.util.Log.d("IdentifyAll", "Detected ${quads.size} cards.")

        // 2. Identification & Logging
        quads.forEachIndexed { index, quad ->
            val corners = quad.toArray().map { p -> Point(p.x / scale, p.y / scale) }
            val fullResQuad = MatOfPoint2f(*corners.toTypedArray())
            
            val unwarper = CardUnwarper()
            val warped = unwarper.unwarp(mat, fullResQuad)
            
            val bmp = Bitmap.createBitmap(warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(warped, bmp)
            
            val identifier = TFLiteCardIdentifier(TFLiteCardFilterModel(InstrumentationRegistry.getInstrumentation().targetContext), TFLiteExpertModel(InstrumentationRegistry.getInstrumentation().targetContext), CardModelMapper.V12, OpenCVWhiteBalancer())
            val result = identifier.identifyCard(bmp)
            
            android.util.Log.d("IdentifyAll", "Card #$index: $result")
            
            // Save for inspection
            saveBitmap(bmp, "chip_$index.jpg")
            
            warped.release()
            fullResQuad.release()
            identifier.close()
        }
        
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

    private fun saveBitmap(bmp: Bitmap, name: String) {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val file = File(appContext.cacheDir, name)
        FileOutputStream(file).use { out ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, out)
        }
        android.util.Log.d("IdentifyAll", "Saved $name to: ${file.absolutePath}")
    }
}

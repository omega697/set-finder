package com.guywithburrito.setfinder.cv

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.json.JSONObject
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.InputStream

/**
 * Chip Extraction Parity Test.
 * Compares Android ChipExtractor output against Python-generated "Gold" references.
 * Proves that Android extraction (RGB) matches Python extraction (BGR) exactly.
 */
@RunWith(AndroidJUnit4::class)
class ChipExtractionParityTest {

    private val extractor = ChipExtractor()

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
    }

    @Test
    fun verifyParityWithPythonReferences() {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val jsonStr = testContext.assets.open("references/extracted_chips/extracted_chips_ground_truth.json").bufferedReader().use { it.readText() }
        val manifest = JSONObject(jsonStr)

        val logBuilder = StringBuilder()
        var totalTested = 0
        var totalPassed = 0

        manifest.keys().forEach { sceneName ->
            val sceneMat = loadFullFrame("scenes/$sceneName")
            val cards = manifest.getJSONArray(sceneName)

            for (i in 0 until cards.length()) {
                val cardData = cards.getJSONObject(i)
                val quadPoints = cardData.getJSONArray("quad")
                val refName = cardData.getString("reference")

                val quad = MatOfPoint2f(
                    Point(quadPoints.getJSONArray(0).getDouble(0), quadPoints.getJSONArray(0).getDouble(1)),
                    Point(quadPoints.getJSONArray(1).getDouble(0), quadPoints.getJSONArray(1).getDouble(1)),
                    Point(quadPoints.getJSONArray(2).getDouble(0), quadPoints.getJSONArray(2).getDouble(1)),
                    Point(quadPoints.getJSONArray(3).getDouble(0), quadPoints.getJSONArray(3).getDouble(1))
                )

                val actualChipBmp = extractor.extract(sceneMat, quad)
                val actualMat = Mat()
                Utils.bitmapToMat(actualChipBmp, actualMat)
                Imgproc.cvtColor(actualMat, actualMat, Imgproc.COLOR_RGBA2RGB)

                val refMat = loadReference("references/extracted_chips/$refName")

                val similarity = calculateSimilarity(actualMat, refMat)
                val psnr = calculatePSNR(actualMat, refMat)
                
                totalTested++
                val pass = psnr > 25.0 
                if (pass) totalPassed++

                logBuilder.append(String.format("%s card %d: PSNR=%.2f, HistCorr=%.4f -> %s\n", 
                    sceneName, i, psnr, similarity, if (pass) "PASS" else "FAIL"))

                actualMat.release(); refMat.release(); quad.release()
            }
            sceneMat.release()
        }

        android.util.Log.i("ChipExtractionParity", "Results:\n$logBuilder")
        android.util.Log.i("ChipExtractionParity", String.format("Passed %d / %d", totalPassed, totalTested))
        
        assertThat(totalPassed).isEqualTo(totalTested)
    }

    private fun loadFullFrame(assetName: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val bitmap = BitmapFactory.decodeStream(testContext.assets.open(assetName))
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }

    private fun loadReference(assetPath: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val bitmap = BitmapFactory.decodeStream(testContext.assets.open(assetPath))
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }

    private fun calculateSimilarity(m1: Mat, m2: Mat): Double {
        val hists = listOf(m1, m2).map { m ->
            val h = Mat()
            Imgproc.calcHist(listOf(m), MatOfInt(0, 1, 2), Mat(), h, MatOfInt(8, 8, 8), MatOfFloat(0f, 256f, 0f, 256f, 0f, 256f))
            Core.normalize(h, h, 0.0, 1.0, Core.NORM_MINMAX)
            h
        }
        return Imgproc.compareHist(hists[0], hists[1], Imgproc.CV_COMP_CORREL)
    }

    private fun calculatePSNR(m1: Mat, m2: Mat): Double {
        val s1 = Mat()
        Core.absdiff(m1, m2, s1)
        s1.convertTo(s1, CvType.CV_32F)
        val s2 = s1.mul(s1)
        val s = Core.sumElems(s2)
        val sse = s.`val`[0] + s.`val`[1] + s.`val`[2]
        if (sse <= 1e-10) return 100.0
        val mse = sse / (m1.channels() * m1.total()).toDouble()
        return 10.0 * Math.log10((255 * 255) / mse)
    }
}

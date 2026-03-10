package com.guywithburrito.setfinder.cv

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.services.storage.TestStorage
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Quad Finding Verification: CardFinder.
 * Measures spatial recall and saves visual debug images for misses.
 */
@RunWith(AndroidJUnit4::class)
class QuadFindingAccuracyTest {

    private lateinit var finder: QuadFinder
    private val testStorage = TestStorage()

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        finder = OpenCVQuadFinder()
    }

    @Test
    fun verifyQuadFinding_SpatialRecall() {
        val logBuilder = StringBuilder()
        var totalCardsExpected = 0
        var totalCardsFound = 0
        var totalCandidatesGenerated = 0
        var scenesTested = 0
        
        QuadFindingGroundTruth.scenes.forEach { (sceneName, gtCards) ->
            scenesTested++
            val mat = loadFullFrame("scenes/$sceneName")
            
            val maxDim = 1000.0
            val scale = maxDim / Math.max(mat.cols().toDouble(), mat.rows().toDouble())
            val small = Mat()
            Imgproc.resize(mat, small, Size(), scale, scale, Imgproc.INTER_AREA)

            val foundCandidates = finder.findCandidatesFull(small)
            totalCandidatesGenerated += foundCandidates.size
            
            val sceneWidth = small.cols().toDouble()
            val sceneHeight = small.rows().toDouble()

            var cardsFoundInScene = 0
            val cardStatus = mutableListOf<Boolean>()

            gtCards.forEachIndexed { index, gtCard ->
                totalCardsExpected++
                val gtPoints = gtCard.pointsPercent.map { p -> Point(p.x * sceneWidth / 100.0, p.y * sceneHeight / 100.0) }
                val gtMat = MatOfPoint2f(*gtPoints.toTypedArray())

                val bestMatch = foundCandidates.map { cand ->
                    val center = cand.quad.getCenter()
                    val centerInside = Imgproc.pointPolygonTest(gtMat, center, false) >= 0
                    val iou = cand.quad.calculateIoU(gtMat)
                    Triple(iou, centerInside, cand.foundBy.joinToString(","))
                }.maxByOrNull { it.first } ?: Triple(0.0, false, "None")

                if (bestMatch.first > 0.4 && bestMatch.second) {
                    cardsFoundInScene++
                    totalCardsFound++
                    cardStatus.add(true)
                } else {
                    cardStatus.add(false)
                    logBuilder.append(String.format("  MISS scene %s: Card #%d (Best IoU: %.2f, CenterInside: %b, FoundBy: %s)\n", 
                        sceneName, index, bestMatch.first, bestMatch.second, bestMatch.third))
                }
                gtMat.release()
            }
            
            // Save visual debug if any misses
            if (cardStatus.any { !it }) {
                drawAndSaveDebug(sceneName, small, gtCards, foundCandidates)
            }

            android.util.Log.i("QuadFindingAccuracy", "$sceneName: Found $cardsFoundInScene/${gtCards.size} cards (Gen ${foundCandidates.size} unique).")
            mat.release(); small.release()
        }

        val finalRecall = if (totalCardsExpected > 0) (totalCardsFound.toFloat() / totalCardsExpected) else 1f
        val avgCandidates = totalCandidatesGenerated.toFloat() / scenesTested
        
        android.util.Log.i("QuadFindingAccuracy", "Results:\n$logBuilder")
        android.util.Log.i("QuadFindingAccuracy", String.format("Overall Spatial Recall: %.2f%% (%d/%d cards found across %d scenes)", 
            finalRecall * 100, totalCardsFound, totalCardsExpected, scenesTested))
        android.util.Log.i("QuadFindingAccuracy", "Average unique candidates per scene: $avgCandidates")

        if (finalRecall < 0.80f) {
            throw AssertionError(String.format("Quad Finding recall too low: %.2f%%. See logcat for misses.", finalRecall * 100))
        }
    }

    private fun drawAndSaveDebug(sceneName: String, small: Mat, gtCards: List<QuadFindingGroundTruth.GTCard>, found: List<CandidateQuad>) {
        val bmp = Bitmap.createBitmap(small.cols(), small.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(small, bmp)
        val canvas = Canvas(bmp)
        
        val gtPaint = Paint().apply { color = android.graphics.Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 3f }
        val foundPaint = Paint().apply { color = android.graphics.Color.RED; style = Paint.Style.STROKE; strokeWidth = 1f; alpha = 128 }
        
        val width = small.cols().toFloat()
        val height = small.rows().toFloat()

        // 1. Draw Found Quads (Red)
        found.forEach { cand ->
            val pts = cand.quad.toArray()
            for (i in 0..3) {
                canvas.drawLine(pts[i].x.toFloat(), pts[i].y.toFloat(), pts[(i+1)%4].x.toFloat(), pts[(i+1)%4].y.toFloat(), foundPaint)
            }
        }

        // 2. Draw Ground Truth (Green)
        gtCards.forEach { gt ->
            val pts = gt.pointsPercent.map { Point(it.x * width / 100.0, it.y * height / 100.0) }
            for (i in 0..3) {
                canvas.drawLine(pts[i].x.toFloat(), pts[i].y.toFloat(), pts[(i+1)%4].x.toFloat(), pts[(i+1)%4].y.toFloat(), gtPaint)
            }
        }

        val outName = "quad_debug_${sceneName.removeSuffix(".jpg")}.jpg"
        testStorage.openOutputFile(outName).use { out ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 90, out)
        }
        bmp.recycle()
    }

    private fun loadFullFrame(assetName: String): Mat {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val bitmap = BitmapFactory.decodeStream(testContext.assets.open(assetName))
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }
}

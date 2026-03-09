package com.guywithburrito.setfinder

import android.graphics.Bitmap
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.cv.OpenCVFrameProcessor
import com.guywithburrito.setfinder.ml.CardIdentifier
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
import org.opencv.android.Utils
import org.opencv.core.*

/**
 * Pure CV/ML pipeline extracted from SetAnalyzer for testability and modularity.
 */
class SetDetector(
    private val finder: CardFinder,
    private val unwarper: CardUnwarper,
    private val identifier: CardIdentifier,
    private val frameProcessor: FrameProcessor = OpenCVFrameProcessor()
) {
    /**
     * Finds and identifies all sets in a single frame.
     */
    fun detectSets(frame: Mat): List<List<SetCard>> {
        val frameWidth = frame.cols().toDouble()
        val frameHeight = frame.rows().toDouble()
        val maxDim = 1000.0
        val scale = maxDim / Math.max(frameWidth, frameHeight)
        
        val analysisFrame = frameProcessor.createMat()
        frameProcessor.resize(frame, analysisFrame, Size(), scale, scale, 1)
        
        // 1. Detect
        val quads = finder.findLikelyCards(analysisFrame)
        val quadPoints = quads.map { it.toList() }
        
        // 2. Identify
        val identified = identifyQuads(frame, quadPoints, scale).filterNotNull()
        
        // 3. Solve
        val sets = findSets(identified)
        
        analysisFrame.release()
        return sets
    }

    /**
     * Identifies a list of quads found in a frame.
     * Quads are provided as List<Point> to avoid native MatOfPoint2f in caller.
     */
    fun identifyQuads(frame: Mat, quadPoints: List<List<Point>>, scale: Double): List<SetCard?> {
        return quadPoints.map { pointsInAnalysisSpace ->
            // Transform coordinates back to full resolution
            val corners = pointsInAnalysisSpace.map { p -> Point(p.x / scale, p.y / scale) }
            val fullResQuad = frameProcessor.createMatOfPoint2f(corners)
            
            val warped = unwarper.unwarp(frame, fullResQuad)
            val bmp = Bitmap.createBitmap(warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(warped, bmp)
            
            val result = identifier.identifyCard(bmp)
            
            warped.release()
            fullResQuad.release()
            result
        }
    }

    /**
     * Finds all valid sets in a list of identified cards.
     */
    fun findSets(cards: List<SetCard>): List<List<SetCard>> {
        val found = mutableListOf<List<SetCard>>()
        for (i in 0 until cards.size) {
            for (j in i + 1 until cards.size) {
                for (k in j + 1 until cards.size) {
                    if (SetCard.isSet(cards[i], cards[j], cards[k])) {
                        found.add(listOf(cards[i], cards[j], cards[k]))
                    }
                }
            }
        }
        return found
    }

    /**
     * Closes the underlying TFLite models.
     */
    fun close() {
        identifier.close()
    }
}

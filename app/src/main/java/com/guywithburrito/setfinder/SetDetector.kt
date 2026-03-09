package com.guywithburrito.setfinder

import android.graphics.Bitmap
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.ChipExtractor
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.cv.OpenCVFrameProcessor
import com.guywithburrito.setfinder.ml.CardIdentifier
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size

/**
 * Pure CV/ML pipeline extracted from SetAnalyzer for testability and modularity.
 * Orchestrates Stage 1 (Extraction), Stage 2 (Filtering), and Stage 3 (Identification).
 */
class SetDetector(
    private val finder: CardFinder,
    private val extractor: ChipExtractor,
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
        
        // 2. Identify (using Stage 1 Extractor and Stages 2/3 Identifier)
        val identified = identifyQuads(frame, quads.map { it.toList() }, scale).filterNotNull()
        
        // 3. Solve
        val sets = findSets(identified)
        
        analysisFrame.release()
        return sets
    }

    /**
     * Identifies a list of quads found in a frame.
     */
    fun identifyQuads(frame: Mat, quadPoints: List<List<Point>>, scale: Double): List<SetCard?> {
        return quadPoints.map { pointsInAnalysisSpace ->
            // Stage 1: Chip Extraction
            val corners = pointsInAnalysisSpace.map { p -> Point(p.x / scale, p.y / scale) }
            val fullResQuad = frameProcessor.createMatOfPoint2f(corners)
            
            val chip = extractor.extract(frame, fullResQuad)
            
            // Stage 2 & 3: Filter & Identify
            val result = identifier.identifyCard(chip)
            
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

package com.guywithburrito.setfinder

import android.graphics.Bitmap
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.QuadFinder
import com.guywithburrito.setfinder.cv.ChipExtractor
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.cv.OpenCVFrameProcessor
import com.guywithburrito.setfinder.ml.CardIdentifier
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size

/**
 * Represents a card detected in a single frame, including its identified 
 * traits and its spatial coordinates.
 */
data class DetectedCard(
    val card: SetCard,
    val bounds: List<Point>
)

/**
 * Pure CV/ML pipeline for detecting and identifying cards in a single image.
 * Provides granular control over detection and identification stages to support 
 * real-time performance optimizations.
 */
class SetDetector(
    private val finder: QuadFinder,
    private val extractor: ChipExtractor,
    private val identifier: CardIdentifier,
    private val frameProcessor: FrameProcessor = OpenCVFrameProcessor()
) {
    /**
     * Convenience method for one-shot detection and identification.
     * Note: This is synchronous and may be slow for large frames.
     */
    fun detectCards(frame: Mat): List<DetectedCard> {
        val frameWidth = frame.cols().toDouble()
        val frameHeight = frame.rows().toDouble()
        val maxDim = 1000.0
        val scale = maxDim / Math.max(frameWidth, frameHeight)
        
        val analysisFrame = frameProcessor.createMat()
        frameProcessor.resize(frame, analysisFrame, Size(), scale, scale, 1)
        
        val quads = detectQuads(analysisFrame)
        val results = mutableListOf<DetectedCard>()
        quads.forEach { pointsInAnalysisSpace ->
            val identified = identifySingleQuad(frame, pointsInAnalysisSpace, scale)
            identified?.let { 
                results.add(DetectedCard(it, pointsInAnalysisSpace))
            }
        }
        
        analysisFrame.release()
        return results
    }

    /**
     * STAGE 1: Fast Geometric Detection.
     * Finds candidate quads in the provided frame.
     */
    fun detectQuads(analysisFrame: Mat): List<List<Point>> {
        return finder.findLikelyCards(analysisFrame).map { it.toList() }
    }

    /**
     * STAGES 2 & 3: Chip Extraction & Identification.
     * Identifies a list of quads found in a frame.
     */
    fun identifyQuads(frame: Mat, quadPoints: List<List<Point>>, scale: Double): List<SetCard?> {
        return quadPoints.map { identifySingleQuad(frame, it, scale) }
    }

    /**
     * Extracts a standardized chip for a given quad. 
     * Useful for persisting identified cards to history.
     */
    fun extractChipForPersistence(frame: Mat, fullResQuad: MatOfPoint2f): Bitmap {
        return extractor.extract(frame, fullResQuad)
    }

    /**
     * Internal helper to extract and identify a single quad.
     */
    private fun identifySingleQuad(frame: Mat, points: List<Point>, scale: Double): SetCard? {
        val corners = points.map { p -> Point(p.x / scale, p.y / scale) }
        val fullResQuad = frameProcessor.createMatOfPoint2f(corners)
        
        val chip = extractor.extract(frame, fullResQuad)
        val result = identifier.identifyCard(chip)
        
        fullResQuad.release()
        return result
    }

    /**
     * Closes the underlying TFLite models.
     */
    fun close() {
        identifier.close()
    }
}

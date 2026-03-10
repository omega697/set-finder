package com.guywithburrito.setfinder.ml

import com.guywithburrito.setfinder.card.SetCard

/**
 * Maps raw TFLite model output indices to SetCard attributes.
 * This class encapsulates the knowledge of which output head corresponds to which trait.
 */
class CardModelMapper(
    private val colorIndex: Int,
    private val countIndex: Int,
    private val patternIndex: Int,
    private val shapeIndex: Int,
    private val colorMap: Map<Int, SetCard.Color>,
    private val shapeMap: Map<Int, SetCard.Shape>,
    private val countMap: Map<Int, SetCard.Count>,
    private val patternMap: Map<Int, SetCard.Pattern>
) {
    /**
     * Translates a map of indexed tensor outputs into a SetCard.
     * Returns null if any required index is missing or if mapping fails.
     */
    fun mapPredictions(predictions: Map<Int, FloatArray>): SetCard? {
        val colIdx = argmax(predictions[colorIndex] ?: return null)
        val cntIdx = argmax(predictions[countIndex] ?: return null)
        val patIdx = argmax(predictions[patternIndex] ?: return null)
        val shpIdx = argmax(predictions[shapeIndex] ?: return null)
        
        return mapIndices(colIdx, shpIdx, cntIdx, patIdx)
    }

    /**
     * Translates raw indices from the model's heads into a SetCard.
     * Returns null if any index is invalid (e.g. index 0 meaning "Background").
     */
    fun mapIndices(colIdx: Int, shpIdx: Int, cntIdx: Int, patIdx: Int): SetCard? {
        val color = colorMap[colIdx] ?: return null
        val shape = shapeMap[shpIdx] ?: return null
        val count = countMap[cntIdx] ?: return null
        val pattern = patternMap[patIdx] ?: return null
        
        return SetCard(shape, pattern, count, color)
    }

    /**
     * Standard argmax for processing raw float arrays from TFLite heads.
     */
    fun argmax(scores: FloatArray): Int {
        if (scores.isEmpty()) return -1
        var bestIdx = 0
        var maxVal = scores[0]
        for (i in 1 until scores.size) {
            if (scores[i] > maxVal) {
                maxVal = scores[i]
                bestIdx = i
            }
        }
        return bestIdx
    }

    companion object {
        /**
         * Canonical mapping for v12 and v13 models.
         * RATIONALE: The order (0=color, 1=count, 2=pattern, 3=shape) is guaranteed 
         * by the 'Clean Slate' model definition in convert_to_tflite.py.
         */
        val V12 = CardModelMapper(
            colorIndex = 0,
            countIndex = 1,
            patternIndex = 2,
            shapeIndex = 3,
            colorMap = mapOf(1 to SetCard.Color.RED, 2 to SetCard.Color.GREEN, 3 to SetCard.Color.PURPLE),
            shapeMap = mapOf(1 to SetCard.Shape.OVAL, 2 to SetCard.Shape.DIAMOND, 3 to SetCard.Shape.SQUIGGLE),
            countMap = mapOf(1 to SetCard.Count.ONE, 2 to SetCard.Count.TWO, 3 to SetCard.Count.THREE),
            patternMap = mapOf(1 to SetCard.Pattern.SOLID, 2 to SetCard.Pattern.SHADED, 3 to SetCard.Pattern.EMPTY)
        )

        // V13 and V14 share the same topology and definition order
        val V13 = CardModelMapper(
            colorIndex = 2,
            countIndex = 0,
            patternIndex = 3,
            shapeIndex = 1,
            colorMap = mapOf(1 to SetCard.Color.RED, 2 to SetCard.Color.GREEN, 3 to SetCard.Color.PURPLE),
            shapeMap = mapOf(1 to SetCard.Shape.OVAL, 2 to SetCard.Shape.DIAMOND, 3 to SetCard.Shape.SQUIGGLE),
            countMap = mapOf(1 to SetCard.Count.ONE, 2 to SetCard.Count.TWO, 3 to SetCard.Count.THREE),
            patternMap = mapOf(1 to SetCard.Pattern.SOLID, 2 to SetCard.Pattern.SHADED, 3 to SetCard.Pattern.EMPTY)
        )
        val V14 = V13
    }
}

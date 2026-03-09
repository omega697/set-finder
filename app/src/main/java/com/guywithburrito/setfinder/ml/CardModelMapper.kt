package com.guywithburrito.setfinder.ml

import com.guywithburrito.setfinder.card.SetCard

/**
 * Maps raw TFLite model output indices to SetCard attributes.
 * This class is 100% logic and can be tested with plain JVM JUnit tests.
 */
class CardModelMapper(
    private val colorMap: Map<Int, SetCard.Color>,
    private val shapeMap: Map<Int, SetCard.Shape>,
    private val countMap: Map<Int, SetCard.Count>,
    private val patternMap: Map<Int, SetCard.Pattern>
) {
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
         * Canonical mapping for v12 "set_card_model_final.tflite".
         */
        val V12 = CardModelMapper(
            colorMap = mapOf(1 to SetCard.Color.RED, 2 to SetCard.Color.GREEN, 3 to SetCard.Color.PURPLE),
            shapeMap = mapOf(1 to SetCard.Shape.OVAL, 2 to SetCard.Shape.DIAMOND, 3 to SetCard.Shape.SQUIGGLE),
            countMap = mapOf(1 to SetCard.Count.ONE, 2 to SetCard.Count.TWO, 3 to SetCard.Count.THREE),
            patternMap = mapOf(1 to SetCard.Pattern.SOLID, 2 to SetCard.Pattern.SHADED, 3 to SetCard.Pattern.EMPTY)
        )

        /**
         * Canonical mapping for v13 "attribute_expert_v13.tflite".
         */
        val V13 = CardModelMapper(
            colorMap = mapOf(1 to SetCard.Color.RED, 2 to SetCard.Color.GREEN, 3 to SetCard.Color.PURPLE),
            shapeMap = mapOf(1 to SetCard.Shape.OVAL, 2 to SetCard.Shape.DIAMOND, 3 to SetCard.Shape.SQUIGGLE),
            countMap = mapOf(1 to SetCard.Count.ONE, 2 to SetCard.Count.TWO, 3 to SetCard.Count.THREE),
            patternMap = mapOf(1 to SetCard.Pattern.SOLID, 2 to SetCard.Pattern.SHADED, 3 to SetCard.Pattern.EMPTY)
        )
    }
}
